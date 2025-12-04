import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel

class DatasetVersion(BaseModel):
    name: str
    version: str
    path: str
    created_at: str
    metadata: Dict[str, Any] = {}

class DataCatalog:
    """
    Manages a catalog of available data versions.
    """
    def __init__(self, catalog_path: str = "data_catalog.json"):
        self.catalog_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), catalog_path)
        self.datasets: Dict[str, List[DatasetVersion]] = self._load_catalog()

    def _load_catalog(self) -> Dict[str, List[DatasetVersion]]:
        if os.path.exists(self.catalog_path):
            try:
                with open(self.catalog_path, "r") as f:
                    data = json.load(f)
                    # Deserialize
                    catalog = {}
                    for name, versions in data.items():
                        catalog[name] = [DatasetVersion(**v) for v in versions]
                    return catalog
            except Exception as e:
                print(f"Error loading catalog: {e}")
                return {}
        return {}

    def _save_catalog(self):
        data = {
            name: [v.dict() for v in versions]
            for name, versions in self.datasets.items()
        }
        with open(self.catalog_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_dataset(self, name: str, version: str, path: str, metadata: Dict[str, Any] = None) -> DatasetVersion:
        """
        Registers a new dataset version.
        """
        if metadata is None:
            metadata = {}
            
        new_version = DatasetVersion(
            name=name,
            version=version,
            path=path,
            created_at=datetime.now().isoformat(),
            metadata=metadata
        )
        
        if name not in self.datasets:
            self.datasets[name] = []
            
        # Check for duplicate version
        for v in self.datasets[name]:
            if v.version == version:
                raise ValueError(f"Version {version} for dataset {name} already exists.")
                
        self.datasets[name].append(new_version)
        self._save_catalog()
        return new_version

    def get_dataset(self, name: str, version: str = "latest") -> Optional[DatasetVersion]:
        """
        Retrieves a dataset version.
        """
        if name not in self.datasets:
            return None
            
        versions = self.datasets[name]
        if not versions:
            return None
            
        if version == "latest":
            # Sort by created_at descending
            sorted_versions = sorted(versions, key=lambda v: v.created_at, reverse=True)
            return sorted_versions[0]
            
        for v in versions:
            if v.version == version:
                return v
        return None

    def list_datasets(self) -> List[str]:
        return list(self.datasets.keys())

# Singleton
_catalog = DataCatalog()

def register_dataset(name: str, version: str, path: str, metadata: Dict[str, Any] = None):
    return _catalog.register_dataset(name, version, path, metadata)

def get_dataset(name: str, version: str = "latest"):
    return _catalog.get_dataset(name, version)
