from pydantic import BaseModel, Field

class LNPInput(BaseModel):
    particle_size_nm: float = Field(..., gt=0, description="Particle size in nanometers")
    ionizable_ratio: float = Field(..., ge=0, le=100, description="Molar ratio percentage")
    helper_ratio: float = Field(..., ge=0, le=100, description="Molar ratio percentage")
    sterol_ratio: float = Field(..., ge=0, le=100, description="Molar ratio percentage")
    peg_ratio: float = Field(..., ge=0, le=100, description="Molar ratio percentage")
    ionizable_lipid: str = Field(..., description="e.g., DLin-MC3-DMA, ALC-0315, SM-102")
    helper_lipid: str = Field(..., description="e.g., DSPC, DOPE")
    sterol_lipid: str = Field(..., description="e.g., Cholesterol")
    peg_lipid: str = Field(..., description="e.g., C14-PEG2000, DMG-PEG2000")
    target_type: str =  Field(..., description="e.g., mRNA, siRNA")

    model_config = {
        "json_schema_extra": {
            "example": {
                "particle_size_nm": 85.5,
                "ionizable_ratio": 50.0,
                "helper_ratio": 10.0,
                "sterol_ratio": 38.5,
                "peg_ratio": 1.5,
                "ionizable_lipid": "DLin-MC3-DMA",
                "peg_lipid": "DMG-PEG2000",
                "sterol_lipid": "Cholesterol",
                "helper_lipid": "DSPC",
                "target_type": "mRNA"
            }
        }
    }

class PredictionResponse(BaseModel):
    predicted_encapsulation_efficiency: float