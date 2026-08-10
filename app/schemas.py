from pydantic import BaseModel

class LNPInput(BaseModel):
    particle_size_nm: float
    ionizable_ratio: float
    helper_ratio: float
    sterol_ratio: float
    peg_ratio: float
    ionizable_lipid: str