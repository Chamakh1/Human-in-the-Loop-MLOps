import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Définir une expérience MLflow
mlflow.set_experiment("iris_rf_experiment")

# Charger les données
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Démarrer un run MLflow
with mlflow.start_run():
    # Créer et entraîner un modèle
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Prédictions et métriques
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    # Log des paramètres et métriques
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    
    # Log du modèle
    mlflow.sklearn.log_model(model, "rf_model")

    print(f"Run terminé, accuracy={acc}")
