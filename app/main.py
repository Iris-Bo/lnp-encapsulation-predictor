from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib

app = FastAPI(
    title="LNP Encapsulation Efficiency Predictor"
)

# Load model

BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(
    BASE_DIR/"models"/"random_forest_baseline.pkl"
)

feature_names = joblib.load(
    BASE_DIR/"models"/"feature_names.pkl"
)

class LNPInput(BaseModel):
    particle_size_nm: float
    ionizable_ratio: float
    helper_ratio: float
    sterol_ratio: float
    peg_ratio: float
    ionizable_lipid: str

#create prediction endpoint
@app.post("/predict")

def predict(data: LNPInput):
    
    # Create row with zeros
    row = {
        feature: 0
        for feature in feature_names
    }

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