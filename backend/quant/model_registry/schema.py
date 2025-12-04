from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ModelConfig(BaseModel):
    """Configuration parameters for a model."""
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters or configuration settings")
    inputs: List[str] = Field(default_factory=list, description="List of input feature names")
    outputs: List[str] = Field(default_factory=list, description="List of output metric names")

class ModelPerformance(BaseModel):
    """Performance metrics for a model version."""
    backtest_id: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict, description="Key performance indicators (Sharpe, RMSE, etc.)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ModelMetadata(BaseModel):
    """Metadata for a registered model."""
    model_id: str = Field(..., description="Unique identifier for the model (e.g., 'val_dcf_growth')")
    version: str = Field(..., description="Semantic version string (e.g., '1.0.0')")
    type: str = Field(..., description="Model type: 'valuation', 'alpha', 'risk', 'ranking'")
    author: str = Field(default="system", description="Author or team responsible")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    description: str = Field(default="", description="Brief description of the model logic")
    status: str = Field(default="dev", description="Lifecycle status: 'dev', 'staging', 'prod', 'archived'")
    
    config: ModelConfig
    performance: Optional[ModelPerformance] = None
    
    # Reproducibility Fields
    training_data_hash: Optional[str] = Field(None, description="Hash of the training data snapshot")
    git_commit_hash: Optional[str] = Field(None, description="Git commit hash of the code used")
    artifact_path: Optional[str] = Field(None, description="Path to serialized model artifact")
    
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
