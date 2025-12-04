import os
import shutil
from datetime import datetime
from typing import Dict, Any
from .catalog import register_dataset

DATA_LAKE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data_lake', 'versioned')

def create_version(
    source_path: str, 
    dataset_name: str, 
    version_tag: str = None, 
    metadata: Dict[str, Any] = None
) -> str:
    """
    Creates an immutable version of a dataset.
    1. Copies source file to versioned storage.
    2. Registers version in DataCatalog.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")
        
    if not os.path.exists(DATA_LAKE_DIR):
        os.makedirs(DATA_LAKE_DIR)
        
    if version_tag is None:
        version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # Define destination path
    # e.g., data_lake/versioned/AAPL_prices/v20231027.parquet
    dataset_dir = os.path.join(DATA_LAKE_DIR, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    filename = os.path.basename(source_path)
    dest_filename = f"{os.path.splitext(filename)[0]}_{version_tag}{os.path.splitext(filename)[1]}"
    dest_path = os.path.join(dataset_dir, dest_filename)
    
    # Copy file (immutable snapshot)
    shutil.copy2(source_path, dest_path)
    
    # Register in Catalog
    register_dataset(dataset_name, version_tag, dest_path, metadata)
    
    return dest_path
