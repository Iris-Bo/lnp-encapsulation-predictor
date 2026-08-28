from fastapi import FastAPI, HTTPException
from pathlib import Path
from contextlib import asynccontextmanager
import pandas as pd
import joblib

from app.schemas import LNPInput, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent

#state dict to hold model artifacts
ml_models = {}

# --- SWAGGER UI DOCUMENTATION ---
api_description = """
Predicts Lipid Nanoparticle encapsulation efficiency based on formulation parameters.

**Valid Formulation Inputs:**

* **Ionizable Lipids:** 4A3-SC8, A2-Iso5-2DC18, ALC-0315, AX4, C12-200, Custom lipid, DLin-KC2-DMA, DLin-MC3-DMA, DLinDMA, DODAP, DOTAP, DOTMA, L202, OF-02, SM-102, cKK-E12, OF-Deg-Lin, 244-cis, 246C10, Heptadecan-9-yl 8-((2-hydroxyethyl)(8-(nonyloxy)-8-oxooctyl)amino)octanoate, OF-C4-Deg-Lin, C9-200, C10-200, BP-Lipid-135, DSDMA, DODMA, DLenDMA
* **Helper Lipids:** DOPC, DOPE, DPPC, DSPC, DSPE, DMPC, DLPE, BMP, SOPC, POPE, 4ME, CL, SOPE, DEPE, DPPE, C16-18:1 PE, POPC, DSPC + TMR-PC
* **PEG Lipids:** ALC-0159, C14-PEG2000, C16-Ceramide-PEG2000, DMG, DMG-PEG2000, DMG-PEG5k, DMPE-PEG2000, DPPE-PEG2000, DSG-PEG2000, DSPE-PEG2000, PEG-c-DMA, PEG-lipid, C8-Ceramide-PEG2000, DMPE-PEG550, DMPE-PEG1000, DPPE-PEG1000, DSPE-PEG550, DSPE-PEG1000, DSPE-2armPEG2000, DOPE-PEG550, DOPE-PEG1000, DOPE-PEG2000, C8-Ceramide-PEG750, C16-Ceramide-PEG750
* **Sterol Lipids:** cholesterol, ¥â-sitosterol, fucosterol, campesterol, stigmastanol
* **Target Types:** ASO, DNA, mRNA, siRNA
"""

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
    lifespan=lifespan,
    description=api_description,
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

    # Pre-process user inputs to lowercase for robust matching
    ionizable_user = data.ionizable_lipid.lower()
    helper_user = data.helper_lipid.lower()
    peg_user = data.peg_lipid.lower()
    sterol_user = data.sterol_lipid.lower()
    target_user = data.target_type.lower()

    # Create mapping dictionaries from the specific sets
    ionizable_map = {k.lower(): k for k in SPECIFIC_IONIZABLE}
    other_ionizable_map = {k.lower(): k for k in OTHER_IONIZABLE}
    
    helper_map = {k.lower(): k for k in SPECIFIC_HELPER}
    other_helper_map = {k.lower(): k for k in OTHER_HELPER}
    
    peg_map = {k.lower(): k for k in SPECIFIC_PEG}
    other_peg_map = {k.lower(): k for k in OTHER_PEG}
    
    sterol_map = {k.lower(): k for k in SPECIFIC_STEROL}
    other_sterol_map = {k.lower(): k for k in OTHER_STEROL}
    
    target_map = {k.lower(): k for k in TARGET_TYPES}
    
    # Ionizable Lipid 
    if ionizable_user in ionizable_map:
        exact_model_name = ionizable_map[ionizable_user]
        row[f"ionizable_{exact_model_name}"] = 1.0
    elif ionizable_user in other_ionizable_map:
        row["ionizable_Other"] = 1.0
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported ionizable lipid.")

    # Helper Lipid 
    if helper_user in helper_map:
        exact_model_name = helper_map[helper_user]
        row[f"helper_{exact_model_name}"] = 1.0
    elif helper_user in other_helper_map:
        row["helper_Other"] = 1.0
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported helper lipid.")

    # PEG Lipid 
    if peg_user in peg_map:
        exact_model_name = peg_map[peg_user]
        row[f"peg_{exact_model_name}"] = 1.0
    elif peg_user in other_peg_map:
        row["peg_Other"] = 1.0
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported PEG lipid.")

    # Sterol Lipid 
    if sterol_user in sterol_map:
        exact_model_name = sterol_map[sterol_user]
        row[f"sterol_{exact_model_name}"] = 1.0
    elif sterol_user not in other_sterol_map:
        raise HTTPException(status_code=400, detail=f"Unsupported sterol lipid.")

    # Target Type 
    if target_user in target_map:
        exact_model_name = target_map[target_user]
        row[f"target_{exact_model_name}"] = 1.0
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported target type.")

    # Predict
    input_df = pd.DataFrame([row])[feature_names]
    prediction = model.predict(input_df)[0]

    return{"predicted_encapsulation_efficiency": round(float(prediction), 2)}

