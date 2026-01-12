# Human-in-the-Loop Annotation & MLOps Platform

![Status](https://img.shields.io/badge/Status-Development-orange)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)

---

## Project Overview

This platform is an interactive image annotation tool enabling a human feedback loop (**Human-in-the-Loop**). It allows for validating AI predictions, correcting bounding boxes, and generating high-quality datasets for model retraining.

### Key Objectives:

* **Portability:** Isolated environment via Docker.
* **AI Audit:** Interface to accept/reject AI agent predictions.
* **MLOps Pipeline:** Integration with MLflow for experiment tracking and DVC for data versioning.

---

## Technical Architecture

| Component            | Technology        | Role                                                            |
| :------------------- | :---------------- | :-------------------------------------------------------------- |
| **Frontend** | JS / HTML5 Canvas | Annotation and review interface                                 |
| **Backend API** | Flask (Python)    | Data management, business logic, AI interaction                 |
| **MLflow** | MLflow Server     | Tracking metrics, parameters, and AI experiment artifacts       |
| **Containerization** | Docker / Compose  | Orchestration and service portability                           |

---

## Installation and Launch

### Prerequisites

* Docker and Docker Compose installed on your machine.
* Git for version control.

### Launch Steps

1. **Clone the project:**

   ```bash
   git clone https://github.com/Chamakh1/Human-in-the-Loop-MLOps.git
   cd Human-in-the-Loop-MLOps

```

2. **Build and start containers:**
```bash
docker-compose up --build

```

3. **Access the application:**
Open your browser at [http://localhost:5000](https://www.google.com/search?q=http://localhost:5000)
4. **Access MLflow:**
MLflow is available at [http://localhost:5001](https://www.google.com/search?q=http://localhost:5001) to track metrics .

---

## Architecture Functioning

### Flask Backend

* Receives annotation data and human feedback via the REST API.
* Interacts with the AI model to generate predictions (`/ai/detect`).
* Saves all information (parameters, metrics, annotated images) into MLflow.
* Dynamically fetches images from `static/tmp` for the interface.

### MLflow

* Each annotation or AI detection session becomes a **run** in MLflow.
* **Parameters** (prompts, frame_id, number of detected objects) are recorded.
* **Metrics**  are tracked.


### Simplified Schema:

```
Frontend (JS/Canvas)
        |
        v
Backend Flask ----------------------> MLflow (tracking server)
  - Receives frames                   - Stores metrics
  - Handles AI detection              - Versioned experiment tracking
  - Saves human feedback              - Enables reproducible training pipelines
        |
        v
/human_study_data

```


## Implemented Features

| Feature | Description |
| --- | --- |
| Dynamic Annotation | Manual drawing of bounding boxes with moving and resizing capabilities. |
| AI Audit | Interface to analyze / accept AI model predictions. |
| Human Feedback | Qualitative rating system (Positive / Negative / Neutral) via keyboard shortcuts (A, S, D). |
| State Management | Action history allowing for undoing and quick error correction. |
| Visual Filtering | Selective hiding or showing of annotations to improve readability. |
| MLOps Tracking w/ MLflow | Recording of metrics, parameters, and artifacts for each run. |

---

## Value of Human-in-the-Loop

Integrating humans into the decision loop brings several major benefits:

* Progressive improvement of annotation quality.
* Rapid detection of AI model errors.
* Creation of a reliable dataset for retraining (based on human feedback).
* Increased confidence in automated decisions.

Human feedback is **strategic data** that transforms a simple annotation application into a true **continuous learning system**.

---

## Roadmap & Future Integrations

### Docker – Containerization

* Ensure portability and reproducibility.
* Flask Backend and MLflow containers.
* Orchestration via `docker-compose`.

### MLflow – Experiment Tracking

* Tracking of model parameters, performance metrics, and human feedback.
* Versioning of artifacts (annotated images, JSON files).

### DVC – Data Management

* Structure and version annotated datasets.
* Management of source images and .pickle annotations without overloading Git.
* Complete traceability for future model retraining.

---

### **Important Note on Moondream VLM**

* This system is optimized for **moondream_v2**, which requires atomic prompting for accurate localization.

   * **One-Word Prompts:** Use single nouns only (e.g., person, tree, bicycle).

   * **Single Object Limit:** The model cannot detect multiple distinct objects simultaneously. For complex scenes, the system performs sequential inference passes per object class.

---

## Conclusion

This platform constitutes a **solid foundation for a modern Human-in-the-Loop system**, combining human interaction, artificial intelligence, and MLOps best practices.

* Portability and reproducibility via Docker.
* Precise tracking and versioning via MLflow.
* Efficient dataset management with DVC.

The project is ready to evolve into a **scalable system oriented towards the continuous improvement of AI agents**.

---
