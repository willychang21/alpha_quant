import os
import yaml
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from quant.model_registry.schema import ModelMetadata

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, registry_path: str = "backend/quant/model_registry/store"):
        """
        Initialize the Model Registry.
        :param registry_path: Path to the directory where model metadata YAMLs are stored.
        """
        # Ensure absolute path if needed, or relative to project root
        if not os.path.isabs(registry_path):
            # Assuming running from project root, but let's be safe
            base_dir = os.getcwd()
            self.registry_path = os.path.join(base_dir, registry_path)
        else:
            self.registry_path = registry_path
            
        if not os.path.exists(self.registry_path):
            os.makedirs(self.registry_path, exist_ok=True)
            
    def register(self, model_id: str, version: str, metadata: Dict[str, Any]) -> bool:
        """
        Register a new model version. Saves metadata to YAML.
        :param model_id: Unique identifier for the model (e.g., 'dcf_v1')
        :param version: Version string (e.g., '1.0.0')
        :param metadata: Dictionary containing model metadata (metrics, parameters, etc.)
        """
        try:
            # Construct ModelMetadata object
            # Ensure required fields are present or set defaults
            meta_obj = ModelMetadata(
                model_id=model_id,
                version=version,
                timestamp=datetime.now(),
                **metadata
            )
            
            filename = f"{model_id}_v{version}.yaml"
            filepath = os.path.join(self.registry_path, filename)
            
            if os.path.exists(filepath):
                logger.warning(f"Model {model_id} v{version} already exists. Overwriting.")
            
            # Convert to dict and dump to YAML
            data = meta_obj.dict()
            
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
                
            logger.info(f"Registered model: {model_id} v{version}")
            return True
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False

    def get_model(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """
        Retrieve model metadata.
        """
        filename = f"{model_id}_v{version}.yaml"
        filepath = os.path.join(self.registry_path, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Model {model_id} v{version} not found.")
            return None
            
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            return ModelMetadata(**data)
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None

    def list_models(self, model_type: Optional[str] = None) -> List[ModelMetadata]:
        """
        List all registered models, optionally filtered by type.
        """
        models = []
        if not os.path.exists(self.registry_path):
            return models
            
        for filename in os.listdir(self.registry_path):
            if filename.endswith(".yaml"):
                filepath = os.path.join(self.registry_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = yaml.safe_load(f)
                    meta = ModelMetadata(**data)
                    
                    if model_type and meta.type != model_type:
                        continue
                        
                    models.append(meta)
                except Exception as e:
                    logger.warning(f"Skipping malformed model file {filename}: {e}")
                    
        return models
