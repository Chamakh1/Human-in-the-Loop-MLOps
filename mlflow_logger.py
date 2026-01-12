import pickle
import mlflow
import os

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("human_in_the_loop_experiment")

possible_paths = [
    "human_study_data/human_data.pickle",
    "Simple-Human-in-the-loop-Annotation-Lab/human_study_data/human_data.pickle"
]

data_path = None
for p in possible_paths:
    if os.path.exists(p):
        data_path = p
        break

if not data_path:
    print(" ERREUR : Fichier introuvable !")
    print(f"Dossier actuel : {os.getcwd()}")
    exit()

print(f"✅ Fichier trouvé : {data_path}")

with open(data_path, "rb") as f:
    data = pickle.load(f)

frame_info = data.get("frame_info", [])
total_frames = len(frame_info)
evaluated = sum(1 for f in frame_info if f.get("is_evaluated", 0) == 1)

try:
    with mlflow.start_run(run_name="manual_upload_success"):
        mlflow.log_metric("total_frames", total_frames)
        mlflow.log_metric("evaluated_frames", evaluated)
        mlflow.log_artifact(data_path)
        print(" ----- RÉUSSITE : La ligne a été ajoutée à MLflow !")
except Exception as e:
    print(f" Erreur MLflow : {e}")