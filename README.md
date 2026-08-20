# LNP Encapsulation Efficiency Predictor

An end-to-end machine learning system to predict lipid nanoparticle (LNP) encapsulation efficiency from formulation parameters, complete with an automated inference API and containerized deployment.

## Tech Stack
- **Modeling & Data:** Python, Scikit-learn, Pandas, NumPy
- **Serving & Validation:** FastAPI, Pydantic, Uvicorn
- **MLOps & DevOps:** Docker, MLflow, GitHub Actions, Pytest

## Project Roadmap

- [x] **Phase 1: Exploratory Data Analysis** — Baseline feature distributions and correlation analysis.
- [x] **Phase 2: Baseline Modeling** — Trained initial Random Forest regressor on formulation parameters.
- [x] **Phase 3: Inference Service** — Built REST API with FastAPI and Pydantic schema validation.
- [ ] **Phase 4: Containerization (In Progress)** — Dockerizing the FastAPI service for reproducible inference.
- [ ] **Phase 5: Automated Testing & CI/CD** — Unit tests (`pytest`) and GitHub Actions build checks.
- [ ] **Phase 6: Advanced Modeling & XAI** — Hyperparameter tuning, XGBoost/LightGBM comparison, and SHAP interpretability.

## Quickstart (Local Inference via Docker)
*(Add run commands here once your container is ready)*