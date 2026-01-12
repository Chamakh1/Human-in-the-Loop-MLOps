# Rapport de Projet : Plateforme d'Annotation et d'Évaluation Human-in-the-Loop

## 1. Présentation du Projet

Ce projet consiste en le développement d’une **application web interactive Human-in-the-Loop (HITL)** dédiée à l’annotation d’images et à l’évaluation de performances d’agents d’intelligence artificielle, en particulier dans le contexte de la **détection d’objets**.

L’objectif principal est d’intégrer efficacement l’expertise humaine dans le cycle de vie d’un système d’IA afin de :

- superviser les prédictions automatiques générées par un modèle de détection,
- corriger et ajuster les erreurs (bounding boxes mal positionnées ou incorrectes),
- valider ou rejeter les décisions de l’agent IA,
- collecter un feedback humain structuré servant à l’amélioration continue du modèle.

Cette approche Human-in-the-Loop permet de réduire les biais du modèle, d’améliorer la qualité des données annotées et de garantir une meilleure fiabilité du système dans des environnements réels.

---

## 2. Architecture Technique Actuelle

L’architecture du projet repose sur une séparation claire entre l’interface utilisateur (Frontend) et la logique applicative (Backend), communiquant via des API REST.

### 2.1 Frontend (Interface Utilisateur)

Le frontend est développé en **HTML5 et JavaScript natif**, en exploitant directement l’API **Canvas** pour le rendu graphique et l’interaction temps réel avec les images.

#### Modes d’interaction

- **Annotate** : création manuelle de boîtes englobantes (*Bounding Boxes*) dessinées par l’utilisateur (couleur rouge).
- **Review** : visualisation, ajustement et validation des prédictions générées automatiquement par l’IA (couleur bleue).

#### Fonctionnalités clés

- Navigation fluide entre les frames (images) avec support du mode automatique (*auto-play*).
- Système robuste de **Undo / Redo** basé sur une pile d’historique des états (*History Stack*).
- Gestion dynamique de la visibilité des annotations (filtrage visuel).
- Conversion précise des coordonnées entre la résolution affichée (canvas) et la résolution réelle des images.
- Interaction souris complète : dessin, déplacement et redimensionnement des bounding boxes.

---

### 2.2 Backend (Serveur API)

Le backend est implémenté en **Python avec Flask**, exposant une API REST accessible localement (`localhost:5000`).  
Il assure la gestion des données, la persistance des annotations et l’interfaçage avec le modèle IA.

#### Endpoints principaux

- `GET /frame/info`  
  Récupération des métadonnées d’une image et des annotations existantes.

- `POST /frame/info`  
  Sauvegarde des annotations créées ou modifiées ainsi que du feedback humain.

- `POST /ai/detect`  
  Déclenchement d’un modèle de détection d’objets (local ou distant) afin de générer des prédictions automatiques.

Le backend agit comme un point central garantissant la cohérence des données entre l’humain et l’agent IA.

---

## 3. Fonctionnalités Implémentées

| Fonctionnalité            | Description |
|---------------------------|-------------|
| Annotation dynamique      | Dessin manuel de bounding boxes avec déplacement et redimensionnement. |
| Audit IA                  | Interface pour analyser, accepter ou rejeter les prédictions du modèle IA. |
| Feedback humain           | Système de notation qualitative (Positive / Négative / Neutre) via raccourcis clavier (A, S, D). |
| Gestion d’état             | Historique des actions permettant l’annulation et la correction rapide des erreurs. |
| Filtrage visuel            | Masquage ou affichage sélectif des annotations pour améliorer la lisibilité. |

---

## 4. Apport du Human-in-the-Loop

L’intégration de l’humain dans la boucle décisionnelle apporte plusieurs bénéfices majeurs :

- amélioration progressive de la qualité des annotations,
- détection rapide des erreurs du modèle IA,
- création d’un jeu de données fiable pour le ré-entraînement,
- augmentation de la confiance dans les décisions automatisées.

Le feedback humain constitue une donnée stratégique permettant de transformer une simple application d’annotation en un véritable **système d’apprentissage continu**.

---

## 5. Fusion Docker + MLflow + Backend

### 5.1 Conteneurisation Docker

La plateforme est entièrement **conteneurisée via Docker**, garantissant portabilité et reproductibilité. Chaque composant (Backend Flask, Base de données, éventuellement moteur IA) fonctionne dans son conteneur isolé.  
Le lancement se fait via `docker-compose`, qui orchestre tous les services.

### 5.2 Suivi des expériences avec MLflow

**MLflow** est intégré pour assurer la **traçabilité des expériences IA et du feedback humain** :

- Chaque prédiction IA et chaque annotation humaine est enregistrée dans MLflow sous forme d’expérience.
- Les métriques suivantes sont suivies :
  - Taux d’acceptation des prédictions IA,
  - Taux de correction des boîtes englobantes,
  - Temps moyen d’annotation par image.
- Les résultats des expériences (annotations, scores, métadonnées) sont stockés dans le **MLflow Tracking Server**, permettant de revenir sur chaque session et de visualiser l’évolution de la performance IA.

### 5.3 Interaction avec le Backend

Le **Backend Flask** agit comme intermédiaire entre les conteneurs Docker et MLflow :

1. L’utilisateur interagit via le frontend.
2. Le backend reçoit les annotations ou déclenche le modèle IA.
3. Les résultats (annotations, prédictions, métriques) sont **automatiquement envoyés à MLflow** pour suivi.
4. Les images et fichiers temporaires restent dans les volumes Docker, garantissant cohérence et isolation.

Cette architecture garantit une **boucle complète Human-in-the-Loop avec traçabilité MLOps**, tout en restant portable et scalable.

---

## 6. Roadmap & Intégrations Futures

### 6.1 Docker – Conteneurisation
- Conteneurisation complète du backend, base de données et moteur IA.
- Lancement via `docker-compose` pour simplifier la mise en place et l’intégration continue.

### 6.2 MLflow – Suivi et Gestion des Modèles
- Suivi des paramètres et métriques pour chaque modèle et annotation.
- Versionnement des modèles via **Model Registry**.

### 6.3 DVC – Gestion et Versionnement des Données
- Versionnement des images et annotations sans surcharger le dépôt Git.
- Historique complet des modifications pour permettre un ré-entraînement futur.

---

## 7. Conclusion

Ce projet constitue une base solide pour une plateforme **Human-in-the-Loop moderne**, combinant interaction humaine, intelligence artificielle, conteneurisation Docker et bonnes pratiques MLOps via MLflow et DVC.  
Il permet la traçabilité complète des annotations, la supervision des modèles IA et l’amélioration continue de la performance des agents.
