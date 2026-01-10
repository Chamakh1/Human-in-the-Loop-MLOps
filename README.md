# Human-in-the-Loop Annotation & MLOps Platform

![Status](https://img.shields.io/badge/Status-Development-orange)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)

##  Présentation du Projet
Cette plateforme est un outil d'annotation d'images interactif permettant une boucle de rétroaction humaine (**Human-in-the-Loop**). Elle permet de valider les prédictions d'IA, de corriger des boîtes englobantes et de générer des jeux de données de haute qualité pour le ré-entraînement de modèles.

### Objectifs Clés :
* **Portabilité :** Environnement isolé via Docker.
* **Audit IA :** Interface pour accepter/rejeter les prédictions de l'agent.
* **Pipeline MLOps :** Intégration future de MLflow et DVC.

---

##  Architecture Technique


| Composant | Technologie | Rôle |
| :--- | :--- | :--- |
| **Frontend** | JS / HTML5 Canvas | Interface d'annotation et révision |
| **Backend API** | Flask (Python) | Gestion des données et logique métier |
| **Base de données**| PostgreSQL | Stockage des métadonnées d'annotation |
| **Conteneurisation**| Docker / Compose | Orchestration des services |

---

##  Installation et Lancement

### Prérequis
* Docker et Docker Compose installés sur votre machine.
* Git pour le versionnement.

### Étapes de lancement
1. **Cloner le projet :**
   ```bash
   git clone <votre-url-repo>
   cd Human-in-the-Loop-MLOps
2. **Construire et lancer les conteneurs :**
       ```bash
   docker-compose up --build

2. **Accéder à l'application :**
     Ouvrez votre navigateur sur http://localhost:5000   
