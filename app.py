import time
import os
import shutil
import pickle
from PIL import Image
import numpy as np 
import moondream as md
import supervision as sv
import mlflow
import cv2
import gym
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS




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

try:
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
    
    fname = str(frame_id) + '.jpg'
    
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
        
        server_data['frame_info'][frame_id]['is_evaluated'] = 1 
        
    else:
        return jsonify({'success': False}), 400, {'ContentType': 'application/json'}
    
    return jsonify({'success': True}), 200, {'ContentType': 'application/json'}

@app.route('/ai/detect', methods=['POST'])
def ai_detect_objects():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 500

    try:
        data = request.get_json()
        frame_id = int(data.get('id', 0))
        object_prompt = data.get('prompt', 'person') 

        img_filename = str(frame_id) + '.jpg'
        img_path = os.path.join(server_data['img_dir'], img_filename)

        if not os.path.exists(img_path):
            return jsonify({'error': 'Image not found'}), 404

        image = Image.open(img_path)

        print(f"[AI] Detecting '{object_prompt}' in frame {frame_id}...")
        result = model.detect(image, object_prompt)

        detections = sv.Detections.from_vlm(
            vlm=sv.VLM.MOONDREAM,
            result=result,
            resolution_wh=image.size
        )

        detected_boxes = []
        for bbox in detections.xyxy:
            x_min, y_min, x_max, y_max = bbox
            
            box = [
                int(y_min), 
                int(y_max), 
                int(x_min), 
                int(x_max)
            ]
            detected_boxes.append(box)

        server_data['frame_info'][frame_id]['ai_predictions'] = detected_boxes

        print(f"[AI] Found {len(detected_boxes)} objects with pixel-accurate coordinates.")
        
        return jsonify({
            'success': True, 
            'ai_boxes': detected_boxes
        })

    except Exception as e:
        print(f'[ERROR] AI Detection failed: {str(e)}')
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/save', methods=['POST'])
def save_human_data():
    try:
        save_dir = os.path.join(app.root_path, "human_study_data")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_file = os.path.join(save_dir, 'human_data.pickle')
        
        with open(save_file, 'wb') as file:
            pickle.dump(server_data, file, protocol=pickle.HIGHEST_PROTOCOL)
            
        print("[INFO] Sauvegarde sur disque réussie.")

        frame_info = server_data.get("frame_info", [])
        total_frames = len(frame_info)
        
        evaluated = sum(1 for f in frame_info if f.get("is_evaluated", 0) == 1)
        
       
        total_human_boxes = 0
        for f in frame_info:
            if 'bounding_boxes' in f and isinstance(f['bounding_boxes'], list):
                total_human_boxes += len(f['bounding_boxes'])

        total_ai_boxes = 0
        for f in frame_info:
            if 'ai_predictions' in f:
                total_ai_boxes += len(f['ai_predictions'])

        progress_percentage = (evaluated / total_frames * 100) if total_frames > 0 else 0

        try:
            with mlflow.start_run(run_name="auto_save_from_backend"):
                mlflow.log_metric("total_frames", total_frames)
                mlflow.log_metric("evaluated_frames", evaluated)
                
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

    img_dir = os.path.join("data", "static", "tmp") 
    data['img_dir'] = img_dir

    if os.path.exists(img_dir):
        shutil.rmtree(img_dir, ignore_errors=True)
    
    os.makedirs(img_dir, exist_ok=True)

    print(f'[INFO] Exporting {data["num_frames"]} images to {img_dir}...')
    
    for idx in range(data['num_frames']):
        img = Image.fromarray(data['rgb_frames'][idx])
        img.save(os.path.join(img_dir, f"{idx}.jpg"), "JPEG")

    return data

def main():
    global server_data
    
    server_data = read_data()
    app.run(debug=True)

if __name__ == '__main__':
    print("[DOCKER] Initialisation des données...")
    server_data = read_data()
    
    if server_data:
        num_frames = server_data.get("num_frames", "N/A")
        img_dir = server_data.get("img_dir", "unknown")

        print(f"[DOCKER] Succès : {num_frames} images générées dans {img_dir}")

    
    app.run(host='0.0.0.0', port=5000, debug=True)