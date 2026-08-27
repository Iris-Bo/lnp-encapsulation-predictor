from fastapi import FastAPI, HTTPException
from pathlib import Path
from contextlib import asynccontextmanager
import pandas as pd
import joblib

from app.schemas import LNPInput, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent

#state dict to hold model artifacts
ml_models = {}

# --- VALIDATION DICTIONARIES ---
# These map exact user string inputs to their correct processing pathway
SPECIFIC_IONIZABLE = {"4A3-SC8", "A2-Iso5-2DC18", "ALC-0315", "AX4", "C12-200", "Custom lipid", "DLin-KC2-DMA", "DLin-MC3-DMA", "DLinDMA", "DODAP", "DOTAP", "DOTMA", "L202", "OF-02", "SM-102"}
OTHER_IONIZABLE = {"cKK-E12", "OF-Deg-Lin", "244-cis", "246C10", "Heptadecan-9-yl 8-((2-hydroxyethyl)(8-(nonyloxy)-8-oxooctyl)amino)octanoate", "OF-C4-Deg-Lin", "C9-200", "C10-200", "BP-Lipid-135", "DSDMA", "DODMA", "DLenDMA"}

SPECIFIC_HELPER = {"DOPC", "DOPE", "DPPC", "DSPC"}
OTHER_HELPER = {"DSPE", "DMPC", "DLPE", "BMP", "SOPC", "POPE", "4ME", "CL", "SOPE", "DEPE", "DPPE", "C16-18:1 PE", "POPC", "DSPC + TMR-PC"}

SPECIFIC_PEG = {"ALC-0159", "C14-PEG2000", "C16-Ceramide-PEG2000", "DMG", "DMG-PEG2000", "DMG-PEG5k", "DMPE-PEG2000", "DPPE-PEG2000", "DSG-PEG2000", "DSPE-PEG2000", "PEG-c-DMA", "PEG-lipid"}
OTHER_PEG = {"C8-Ceramide-PEG2000", "DMPE-PEG550", "DMPE-PEG1000", "DPPE-PEG1000", "DSPE-PEG550", "DSPE-PEG1000", "DSPE-2armPEG2000", "DOPE-PEG550", "DOPE-PEG1000", "DOPE-PEG2000", "C8-Ceramide-PEG750", "C16-Ceramide-PEG750"}

SPECIFIC_STEROL = {"cholesterol"}
OTHER_STEROL = {"¥â-sitosterol", "fucosterol", "campesterol", "stigmastanol"}

TARGET_TYPES = {"ASO", "DNA", "mRNA", "siRNA"}

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

    # Map numerical features
    row["particle_size_nm"] = data.particle_size_nm
    row["ionizable_ratio"] = data.ionizable_ratio
    row["helper_ratio"] = data.helper_ratio
    row["sterol_ratio"] = data.sterol_ratio
    row["peg_ratio"] = data.peg_ratio

    # Map one-hot encoded categorical features
    
    ## Ionizable Lipid
    if data.ionizable_lipid in SPECIFIC_IONIZABLE:
        row[f"ionizable_{data.ionizable_lipid}"] = 1.0
    elif data.ionizable_lipid in OTHER_IONIZABLE:
        row["ionizable_Other"] = 1.0
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported ionizable lipid: {data.ionizable_lipid}. Check documentation for valid options.")

    ## Helper Lipid
    if data.helper_lipid in SPECIFIC_HELPER:
        row[f"helper_{data.helper_lipid}"] = 1.0
    elif data.helper_lipid in OTHER_HELPER:
        row["helper_Other"] = 1.0
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported helper lipid: {data.helper_lipid}. Check documentation for valid options.")

    ## PEG Lipid
    if data.peg_lipid in SPECIFIC_PEG:
        row[f"peg_{data.peg_lipid}"] = 1.0
    elif data.peg_lipid in OTHER_PEG:
        row["peg_Other"] = 1.0
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported PEG lipid: {data.peg_lipid}. Check documentation for valid options.")

    ## Sterol Lipid (Note: Non-cholesterol valid sterols safely remain mapped to 0)
    if data.sterol_lipid in SPECIFIC_STEROL:
        row[f"sterol_{data.sterol_lipid}"] = 1.0
    elif data.sterol_lipid not in OTHER_STEROL:
        raise HTTPException(status_code=400, detail=f"Unsupported sterol lipid: {data.sterol_lipid}. Check documentation for valid options.")

    ## Target Type
    if data.target_type in TARGET_TYPES:
        row[f"target_{data.target_type}"] = 1.0
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported target type: {data.target_type}. Check documentation for valid options.")

    # Predict
    input_df = pd.DataFrame([row])
    prediction = model.predict(input_df)[0]

    return{"predicted_encapsulation_efficiency": round(float(prediction), 2)}

