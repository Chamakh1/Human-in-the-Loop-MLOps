import time
import os
import shutil
import pickle
from PIL import Image
import numpy as np # Added numpy just in case

# --- NEW: Import Moondream ---
import moondream as md
import supervision as sv

import cv2
import gym
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
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
    target_dir = os.path.join(app.static_folder, "tmp")
    
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
        server_data['frame_info'][frame_id].update(new_info)
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
        save_dir = os.path.join(app.root_path, "human_study_data")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_file = os.path.join(save_dir, 'human_data.pickle')
        with open(save_file, 'wb') as file:
            pickle.dump(server_data, file, protocol=pickle.HIGHEST_PROTOCOL)
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
        img_dir = os.path.join(app.static_folder, "tmp")
        return {
            'rgb_frames': [np.zeros((210, 160, 3), dtype=np.uint8)],
            'actions': [0],
            'frame_info': [{'bounding_boxes': [], 'ai_predictions': [], 'human_feedback': 0, 'is_evaluated': 0}],
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
    img_dir = os.path.join(app.static_folder, "tmp")
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
        print(f"[DOCKER] Succès : {server_data['num_frames']} images générées dans {server_data['img_dir']}")
    
    # 3. Lancer Flask sur 0.0.0.0 pour Docker
    app.run(host='0.0.0.0', port=5000, debug=True)