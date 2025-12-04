import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

SNAPSHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_snapshots')

class DataSnapshot:
    """
    Manages data snapshots for reproducible model runs.
    """
    def __init__(self, snapshots_dir: str = SNAPSHOTS_DIR):
        self.snapshots_dir = snapshots_dir
        if not os.path.exists(self.snapshots_dir):
            os.makedirs(self.snapshots_dir)

    def create_snapshot(self, data_path: str, description: str = "") -> str:
        """
        Creates a snapshot record for a given data file.
        Returns the SHA256 hash of the file.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Compute hash
        sha256_hash = hashlib.sha256()
        with open(data_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        file_hash = sha256_hash.hexdigest()
        
        # Store metadata
        metadata = {
            "hash": file_hash,
            "original_path": data_path,
            "created_at": datetime.now().isoformat(),
            "description": description
        }
        
        metadata_path = os.path.join(self.snapshots_dir, f"{file_hash}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        return file_hash

    def get_snapshot(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves snapshot metadata by hash.
        """
        metadata_path = os.path.join(self.snapshots_dir, f"{file_hash}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

# Singleton
_snapshot_manager = DataSnapshot()

def create_snapshot(data_path: str, description: str = "") -> str:
    return _snapshot_manager.create_snapshot(data_path, description)

def get_snapshot(file_hash: str) -> Optional[Dict[str, Any]]:
    return _snapshot_manager.get_snapshot(file_hash)
