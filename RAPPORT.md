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
| Feedback humain            | Système de notation qualitative (Positive / Négative / Neutre) via raccourcis clavier (A, S, D). |
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

## 5. Roadmap & Intégrations Futures

L’évolution du projet vise à renforcer sa robustesse, sa traçabilité et son passage à l’échelle via des pratiques MLOps.

### 5.1 Docker – Conteneurisation

**Objectif :** garantir la portabilité et la reproductibilité de l’application.

**Implémentation prévue :**
- Création d’un `Dockerfile` pour le backend Flask.
- Utilisation de `docker-compose.yml` pour orchestrer les services (API, base de données, moteur d’inférence IA).

---

### 5.2 MLflow – Suivi et Gestion des Modèles

**Objectif :** assurer un suivi précis des expériences et des performances des modèles IA.

**Implémentation prévue :**
- Enregistrement des paramètres des modèles et des prompts.
- Suivi des métriques issues du feedback humain (taux d’acceptation, taux de correction).
- Versionnement et gestion des modèles via un **Model Registry**.

---

### 5.3 DVC – Gestion et Versionnement des Données

**Objectif :** structurer et versionner les jeux de données et annotations.

**Implémentation prévue :**
- Gestion des images sources sans surcharge du dépôt Git.
- Versionnement des fichiers JSON d’annotations.
- Traçabilité complète des données (*data lineage*) pour le ré-entraînement futur des modèles.

---

## 6. Conclusion

Ce projet constitue une base solide pour une plateforme **Human-in-the-Loop moderne**, combinant interaction humaine, intelligence artificielle et bonnes pratiques MLOps.  
Il ouvre la voie à un système évolutif, fiable et orienté vers l’amélioration continue des performances des agents IA.
