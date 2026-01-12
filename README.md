# Human-in-the-Loop Annotation & MLOps Platform

![Status](https://img.shields.io/badge/Status-Development-orange)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)

---

## Présentation du Projet

Cette plateforme est un outil d'annotation d'images interactif permettant une boucle de rétroaction humaine (**Human-in-the-Loop**). Elle permet de valider les prédictions d'IA, de corriger des boîtes englobantes et de générer des jeux de données de haute qualité pour le ré-entraînement de modèles.

### Objectifs Clés :

* **Portabilité :** environnement isolé via Docker.
* **Audit IA :** interface pour accepter/rejeter les prédictions de l’agent IA.
* **Pipeline MLOps :** intégration avec MLflow pour le suivi des expériences et DVC pour le versionnement des données.

---

## Architecture Technique

| Composant            | Technologie       | Rôle                                                            |
| :------------------- | :---------------- | :-------------------------------------------------------------- |
| **Frontend**         | JS / HTML5 Canvas | Interface d'annotation et révision                              |
| **Backend API**      | Flask (Python)    | Gestion des données, logique métier, interaction avec IA        |
| **MLflow**           | MLflow Server     | Suivi des métriques, paramètres et artefacts des expériences IA |
| **Base de données**  | PostgreSQL        | Stockage des métadonnées d'annotation                           |
| **Conteneurisation** | Docker / Compose  | Orchestration et portabilité des services                       |

---

## Installation et Lancement

### Prérequis

* Docker et Docker Compose installés sur votre machine.
* Git pour le versionnement.

### Étapes de lancement

1. **Cloner le projet :**

   ```bash
   git clone <votre-url-repo>
   cd Human-in-the-Loop-MLOps
   ```

2. **Construire et lancer les conteneurs :**

   ```bash
   docker-compose up --build
   ```

3. **Accéder à l'application :**
   Ouvrez votre navigateur sur [http://localhost:5000](http://localhost:5000)

4. **Accéder à MLflow :**
   MLflow est disponible sur [http://localhost:5001](http://localhost:5001) pour suivre les métriques et artefacts.

---

## Fonctionnement de l’Architecture

### Backend Flask

* Reçoit les données d’annotation et les feedbacks humains via l’API REST.
* Interagit avec le modèle IA pour générer des prédictions (`/ai/detect`).
* Sauvegarde toutes les informations (paramètres, métriques, images annotées) dans MLflow.
* Génère dynamiquement les images dans `static/tmp` pour l’interface.

### MLflow

* Chaque session d’annotation ou détection IA devient une **run** dans MLflow.
* Les **paramètres** (prompts, frame_id, nombre d’objets détectés) sont enregistrés.
* Les **métriques** (feedback humain, corrections, précision) sont suivies.
* Les **artefacts** (images annotées, JSON des annotations) sont stockés pour réutilisation.

### Schéma simplifié :

```
Frontend (JS/Canvas)
        |
        v
Backend Flask ----------------------> MLflow (tracking server)
  - Receives frames                   - Stores metrics, params, artifacts
  - Handles AI detection              - Versioned experiment tracking
  - Saves human feedback              - Enables reproducible training pipelines
        |
        v
Static/tmp (Images)
```

---

## Fonctionnalités Implémentées

| Fonctionnalité          | Description                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| Annotation dynamique    | Dessin manuel de bounding boxes avec déplacement et redimensionnement.                           |
| Audit IA                | Interface pour analyser, accepter ou rejeter les prédictions du modèle IA.                       |
| Feedback humain         | Système de notation qualitative (Positive / Négative / Neutre) via raccourcis clavier (A, S, D). |
| Gestion d’état          | Historique des actions pour annulation et correction rapide des erreurs.                         |
| Filtrage visuel         | Masquage ou affichage sélectif des annotations pour améliorer la lisibilité.                     |
| Suivi MLOps avec MLflow | Enregistrement des métriques, paramètres et artefacts pour chaque run.                           |

---

## Apport du Human-in-the-Loop

L’intégration de l’humain dans la boucle décisionnelle apporte plusieurs bénéfices :

* Amélioration progressive de la qualité des annotations.
* Détection rapide des erreurs du modèle IA.
* Création d’un jeu de données fiable pour le ré-entraînement.
* Augmentation de la confiance dans les décisions automatisées.

Le feedback humain est **une donnée stratégique** permettant de transformer une simple application d’annotation en un **système d’apprentissage continu**.

---

## Roadmap & Intégrations Futures

### Docker – Conteneurisation

* Garantir la portabilité et la reproductibilité.
* Conteneur backend Flask et MLflow.
* Orchestration via `docker-compose`.

### MLflow – Suivi des Expériences

* Suivi des paramètres du modèle, métriques de performance et feedback humain.
* Versionnement des artefacts (images annotées, fichiers JSON).

### DVC – Gestion des Données

* Structurer et versionner les jeux de données annotés.
* Gestion des images sources et des annotations JSON sans surcharger Git.
* Traçabilité complète pour le ré-entraînement futur des modèles.

---

## Conclusion

Cette plateforme constitue une **base solide pour un système Human-in-the-Loop moderne**, combinant interaction humaine, intelligence artificielle et bonnes pratiques MLOps.

* Portabilité et reproductibilité via Docker.
* Suivi et versionnement précis via MLflow.
* Gestion efficace des jeux de données avec DVC.

Le projet est prêt à évoluer vers un **système évolutif et orienté amélioration continue des agents IA**.
