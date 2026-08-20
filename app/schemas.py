from pydantic import BaseModel, Field

class LNPInput(BaseModel):
    particle_size_nm: float = Field(..., gt=0, description="Particle size in nanometers")
    ionizable_ratio: float = Field(..., ge=0, le=100, description="Molar ratio percentage")
    helper_ratio: float = Field(..., ge=0, le=100, description="Molar ratio percentage")
    sterol_ratio: float = Field(..., ge=0, le=100, description="Molar ratio percentage")
    peg_ratio: float = Field(..., ge=0, le=100, description="Molar ratio percentage")
    ionizable_lipid: str = Field(..., description="e.g., MC3, ALC-0315, SM-102")

    model_config = {
        "json_schema_extra": {
            "example": {
                "particle_size_nm": 85.5,
                "ionizable_ratio": 50.0,
                "helper_ratio": 10.0,
                "sterol_ratio": 38.5,
                "peg_ratio": 1.5,
                "ionizable_lipid": "MC3"
            }
        }
    }

class PredictionResponse(BaseModel):
    predicted_encapsulation_efficiency: float