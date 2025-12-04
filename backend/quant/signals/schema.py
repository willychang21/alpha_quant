from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class SignalBase(BaseModel):
    ticker: str
    model_name: str
    model_version: str
    score: float
    metadata_json: Optional[str] = None
    timestamp: datetime

class SignalCreate(SignalBase):
    pass

class SignalResponse(SignalBase):
    id: int

    class Config:
        orm_mode = True
