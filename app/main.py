from fastapi import FastAPI
from pathlib import Path
from contextlib import asynccontextmanager
import pandas as pd
import joblib

from app.schemas import LNPInput, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent

#state dict to hold model artifacts
ml_models = {}

@asynccontextmanager
async def lifespan(app:FastAPI):
    # Load artifacts on startup
    ml_models["model"] = joblib.load(BASE_DIR/"models"/"random_forest_baseline.pkl")
    ml_models["feature_names"]=joblib.load(BASE_DIR/"models"/"feature_names.pkl")
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(
    title="LNP Encapsulation Efficiency Predictor",
    lifespan=lifespan
)

#Get endpoint to verify if service is running
@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "healthy"}

#create prediction endpoint
@app.post("/predict",response_model=PredictionResponse, tags=["Inference"])

def predict(data: LNPInput):
    model = ml_models["model"]
    feature_names = ml_models["feature_names"]

    # Create row with zeros
    row = {feature: 0 for feature in feature_names}

    # Numerical features
    row["particle_size_nm"] = data.particle_size_nm
    row["ionizable_ratio"] = data.ionizable_ratio
    row["helper_ratio"] = data.helper_ratio
    row["sterol_ratio"] = data.sterol_ratio
    row["peg_ratio"] = data.peg_ratio

    # One-hot encoded lipid
    lipid_column = f"ionizable_{data.ionizable_lipid}"

    if lipid_column in row:
        row[lipid_column] = 1
    else:
        row["ionizable_Other"] = 1 #adapt this later on, so no 'unknown' lipids can be entered as input

    input_df = pd.DataFrame([row])

    prediction = model.predict(input_df)[0]

    return{"predicted_encapsulation_efficiency": round(float(prediction), 2)}

