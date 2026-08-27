# LNP Encapsulation Efficiency Predictor

An end-to-end machine learning system to predict lipid nanoparticle (LNP) encapsulation efficiency from formulation parameters, complete with an automated inference API and containerized deployment.
The LNP Atlas dataset is used ("https://lnp-atlas.kisti.re.kr/"). 

## Tech Stack
- **Modeling & Data:** Python, Scikit-learn, Pandas, NumPy
- **Serving & Validation:** FastAPI, Pydantic, Uvicorn
- **MLOps & DevOps:** Docker, MLflow, GitHub Actions, Pytest

## Project Roadmap

- [x] **Phase 1: Exploratory Data Analysis** — Baseline feature distributions and correlation analysis.
- [x] **Phase 2: Baseline Modeling** — Trained initial Random Forest regressor on formulation parameters.
- [x] **Phase 3: Inference Service** — Built REST API with FastAPI and Pydantic schema validation.
- [x] **Phase 4: Containerization** — Dockerized FastAPI service for reproducible inference.
- [x] **Phase 5: Automated Testing & CI/CD** — Unit tests (`pytest`) and GitHub Actions build checks.
- [x] **Phase 6: Experiment Tracking & MLOps** — Integrated MLflow to log hyperparameters, performance metrics, and model artifacts.
- [ ] **Phase 7: Model improvement and comparison** — Addition of features, hyperparameter tuning, XGBoost comparison, and SHAP interpretability.

## Quickstart

### Option 1: Run via Docker (Recommended)

1. **Build the container image:**
   ```bash
   docker build -t lnp-prediction .
   ```

2. **Run the container:**
   ```bash
   docker run -d -p 8000:8000 --name lnp-api lnp-prediction
   ```

3. **Access the API:**
   - Interactive Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Health Check: [http://localhost:8000/health](http://localhost:8000/health)

4. **Stop and remove the container:**
   ```bash
   docker stop lnp-api; docker rm lnp-api
   ```

---

### Option 2: Local Development Setup (API)

1. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell):
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install development dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Run the FastAPI server locally:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

4. **Access the local API:**
   - Interactive Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### Option 3: Model Training and Experiment Tracking (MLflow)

To reproduce the model training, run hyperparameter tuning, or view the experiment metrics, use the local MLflow tracking server alongside Jupyter.

1. **Activate your virtual environment and install development dependencies (See Option 2)**

2. **Start the MLflow UI:**
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

3. **View the Dashboard:**

   Open http://127.0.0.1:5000 in your browser to explore the model registry, metric comparisons, and artifact logs.

4. **Run the Notebooks:**

   Open a new terminal, activate the environment, and start Jupyter to interact with the training code:
    ```bash
   jupyter notebook
   ```
---

## Example API Request

**POST** `/predict`

```json
{
  "particle_size_nm": 85.5,
  "ionizable_ratio": 50.0,
  "helper_ratio": 10.0,
  "sterol_ratio": 38.5,
  "peg_ratio": 1.5,
  "ionizable_lipid": "DLin-MC3-DMA",
  "helper_lipid": "DSPC",
  "sterol_lipid": "Cholesterol",
  "peg_lipid": "DMG-PEG2000",
  "target_type": "mRNA"
}
```

**Response:**
```json
{
  "predicted_encapsulation_efficiency": 92.4
}
```
