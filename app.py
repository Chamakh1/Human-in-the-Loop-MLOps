import time
import os
import shutil
import pickle
from PIL import Image
import numpy as np # Added numpy just in case

# --- NEW: Import Moondream ---
import moondream as md
import supervision as sv
import mlflow
import cv2
import gym
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS




# --- CONFIGURATION MLFLOW ---
# On récupère l'adresse définie dans le docker-compose (http://mlflow:5000)
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("human_in_the_loop_experiment")
print(f"[INFO] MLflow connecté à : {mlflow_uri}")
app = Flask(
    __name__,
    static_folder="data/static",
    static_url_path="/static"
)
CORS(app)
# --- CONFIGURATION ---
# 1. Set your NEW API key here (or better, in your OS environment variables)
# os.environ["MOONDREAM_API_KEY"] = "paste_your_new_key_here"

# 2. Initialize Moondream
# We wrap this in a try/except so the app doesn't crash if the key is missing during dev
try:
    # md.vl checks for MOONDREAM_API_KEY env variable automatically if passed, 
    # or you can pass api_key="..." directly.
    model = md.vl(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiIyYmFiNmMxYi1iZTliLTRkMWQtYjI0Yi04NTYyOWUyMmRkN2MiLCJvcmdfaWQiOiI0V2I3dUdEbW11eml2RWdTQnZ4UnhrajFPa2d0anptZCIsImlhdCI6MTc2Nzk2MjkwMCwidmVyIjoxfQ.PRYhyjlHRG0JlEbR7JUOVt3Md6NDlwo0nGerJvQGCoA")
    print("[INFO] Moondream model initialized successfully.")
except Exception as e:
    print(f"[WARNING] Moondream failed to load: {e}")
    model = None

server_data = None
data_file = os.path.join(app.static_folder, 'asterix', 'images.pkl')


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/frame')
def get_frame():
    """Get single frame"""
    frame_id = request.args.get('id', default=0, type=int)
    
    # On force le nom du fichier
    fname = str(frame_id) + '.jpg'
    
    # On définit le chemin absolu vers le dossier tmp de Flask
    # app.static_folder pointe vers le dossier 'static' du projet
    target_dir = os.path.join("data", "static", "tmp")
    
    return send_from_directory(target_dir, fname)


@app.route('/frame/info', methods=['GET'])
def get_frame_info():
    """Get single frame"""
    frame_id = request.args.get('id')
    if frame_id is None:
        frame_id = 0
    frame_id = int(frame_id)

    frame_info = server_data['frame_info'][frame_id]
    
    # Ensure ai_predictions key exists in the response
    if 'ai_predictions' not in frame_info:
        frame_info['ai_predictions'] = []

    frame_info.update({
        'num_frames': server_data['num_frames'],
        'frame_id': frame_id,
        'rgb_obs_height': server_data['rgb_obs_shape'][0],
        'rgb_obs_width': server_data['rgb_obs_shape'][1],
        'action': server_data['actions'][frame_id]
    })
    return jsonify(frame_info)


@app.route('/frame/info', methods=['POST'])
def update_frame_info():
    new_info = request.get_json()
    frame_id = request.args.get('id')

    if frame_id is not None and new_info is not None:
        frame_id = int(frame_id)
        global server_data
        
        # 1. On met à jour les infos (les boîtes)
        server_data['frame_info'][frame_id].update(new_info)
        
        # 2. --- LA LIGNE MANQUANTE ---
        # On force le statut à "évalué" (1) pour que MLflow le compte
        server_data['frame_info'][frame_id]['is_evaluated'] = 1 
        # -----------------------------
        
    else:
        return jsonify({'success': False}), 400, {'ContentType': 'application/json'}
    
    return jsonify({'success': True}), 200, {'ContentType': 'application/json'}

# --- NEW: AI Detection Route ---
@app.route('/ai/detect', methods=['POST'])
def ai_detect_objects():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 500

    try:
        data = request.get_json()
        frame_id = int(data.get('id', 0))
        object_prompt = data.get('prompt', 'person') 

        # 1. Locate and open the image
        img_filename = str(frame_id) + '.jpg'
        img_path = os.path.join(server_data['img_dir'], img_filename)

        if not os.path.exists(img_path):
            return jsonify({'error': 'Image not found'}), 404

        image = Image.open(img_path)

        # 2. Run Moondream Detection
        print(f"[AI] Detecting '{object_prompt}' in frame {frame_id}...")
        result = model.detect(image, object_prompt)

        # 3. Use Supervision to scale coordinates from [0-1] to [pixels]
        # Supervision expects resolution as (width, height) which is image.size
        detections = sv.Detections.from_vlm(
            vlm=sv.VLM.MOONDREAM,
            result=result,
            resolution_wh=image.size
        )

        # 4. Convert Supervision format to your App's format
        # sv.Detections.xyxy returns: [x_min, y_min, x_max, y_max]
        # Your App expects: [y_min, y_max, x_min, x_max]
        detected_boxes = []
        for bbox in detections.xyxy:
            x_min, y_min, x_max, y_max = bbox
            
            # Re-ordering for your frontend logic
            box = [
                int(y_min), 
                int(y_max), 
                int(x_min), 
                int(x_max)
            ]
            detected_boxes.append(box)

        # 5. Save AI predictions in server_data
        server_data['frame_info'][frame_id]['ai_predictions'] = detected_boxes

        print(f"[AI] Found {len(detected_boxes)} objects with pixel-accurate coordinates.")
        
        return jsonify({
            'success': True, 
            'ai_boxes': detected_boxes
        })

    except Exception as e:
        print(f'[ERROR] AI Detection failed: {str(e)}')
        return jsonify({'success': False, 'error': str(e)}), 500
    
# noinspection PyBroadException
@app.route('/save', methods=['POST'])
def save_human_data():
    try:
        # 1. Sauvegarde Classique sur le Disque
        save_dir = os.path.join(app.root_path, "human_study_data")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_file = os.path.join(save_dir, 'human_data.pickle')
        
        with open(save_file, 'wb') as file:
            pickle.dump(server_data, file, protocol=pickle.HIGHEST_PROTOCOL)
            
        print("[INFO] Sauvegarde sur disque réussie.")

        # 2. ENVOI AUTOMATIQUE VERS MLFLOW
        # --- CALCUL DES NOUVELLES METRIQUES ---
        frame_info = server_data.get("frame_info", [])
        total_frames = len(frame_info)
        
        # Compte les images terminées
        evaluated = sum(1 for f in frame_info if f.get("is_evaluated", 0) == 1)
        
        # Compte le nombre total de boîtes dessinées par l'humain
       
        total_human_boxes = 0
        for f in frame_info:
            # On regarde la clé standard 'bounding_boxes' où sont stockées les validations
            if 'bounding_boxes' in f and isinstance(f['bounding_boxes'], list):
                total_human_boxes += len(f['bounding_boxes'])

        # Compte le nombre total de boîtes trouvées par l'IA (Moondream)
        total_ai_boxes = 0
        for f in frame_info:
            if 'ai_predictions' in f:
                total_ai_boxes += len(f['ai_predictions'])

        # Calcul du pourcentage de progression (0 à 100)
        progress_percentage = (evaluated / total_frames * 100) if total_frames > 0 else 0

        # --- ENVOI A MLFLOW ---
        try:
            with mlflow.start_run(run_name="auto_save_from_backend"):
                # Métriques existantes
                mlflow.log_metric("total_frames", total_frames)
                mlflow.log_metric("evaluated_frames", evaluated)
                
                # NOUVELLES Métriques
                mlflow.log_metric("human_boxes_count", total_human_boxes)
                mlflow.log_metric("ai_boxes_count", total_ai_boxes)
                mlflow.log_metric("progress_percentage", progress_percentage)
                
                mlflow.log_artifact(save_file)
                print("***************[MLFLOW] Données complètes envoyées avec succès !")
        except Exception as e_mlflow:
            print(f"[WARNING] Sauvegardé sur disque, mais échec MLflow : {e_mlflow}")

    except Exception as e:
        print('[ERROR] Fail to save server data:', str(e))
        return jsonify({'success': False}), 400, {'ContentType': 'application/json'}
    
    return jsonify({'success': True}), 200, {'ContentType': 'application/json'}
def read_data():
    """Load data on server"""
    print('[INFO] Loading server resources ...')
    
    if not os.path.exists(data_file):
        print("[WARNING] Data file not found. Creating dummy data.")
        # Utilisation de os.path.join pour la compatibilité Linux/Windows
        img_dir = os.path.join("data", "static", "tmp")
        dummy_frame = np.zeros((210, 160, 3), dtype=np.uint8)
        return {
            'rgb_frames': [dummy_frame],
            'rgb_obs_shape': dummy_frame.shape,
            'num_frames': 1,
            'actions': [0],
            'frame_info': [{
                'bounding_boxes': [],
                'ai_predictions': [],
                'human_feedback': 0,
                'is_evaluated': 0
            }],
            'img_dir': img_dir
        }

    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    data['rgb_obs_shape'] = data['rgb_frames'][0].shape
    data['num_frames'] = len(data['rgb_frames'])
    
    if 'frame_info' not in data:
        data['frame_info'] = [{'bounding_boxes': [], 'human_feedback': 0, 'is_evaluated': 0} for _ in range(data['num_frames'])]
    
    for info in data['frame_info']:
        if 'ai_predictions' not in info:
            info['ai_predictions'] = []

    # --- SECTION CRITIQUE : GÉNÉRATION DES IMAGES ---
    img_dir = os.path.join("data", "static", "tmp") 
    data['img_dir'] = img_dir

    # Création propre du dossier
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir, ignore_errors=True)
    
    # exist_ok=True évite les erreurs si le dossier est créé entre-temps
    os.makedirs(img_dir, exist_ok=True)

    print(f'[INFO] Exporting {data["num_frames"]} images to {img_dir}...')
    
    for idx in range(data['num_frames']):
        img = Image.fromarray(data['rgb_frames'][idx])
        img.save(os.path.join(img_dir, f"{idx}.jpg"), "JPEG")

    return data

def main():
    global server_data
    # Set the key here if not in environment variables
    # os.environ["MOONDREAM_API_KEY"] = "YOUR_NEW_KEY_HERE"
    
    server_data = read_data()
    app.run(debug=True)

if __name__ == '__main__':
    # 1. Charger les données et générer les images AVANT de lancer le serveur
    print("[DOCKER] Initialisation des données...")
    server_data = read_data()
    
    # 2. Vérification rapide
    if server_data:
        num_frames = server_data.get("num_frames", "N/A")
        img_dir = server_data.get("img_dir", "unknown")

        print(f"[DOCKER] Succès : {num_frames} images générées dans {img_dir}")

    
    # 3. Lancer Flask sur 0.0.0.0 pour Docker
    app.run(host='0.0.0.0', port=5000, debug=True)