from quant.data.versioning import create_version
from quant.data.catalog import get_dataset
import os
import shutil

def test_data_layer():
    # 1. Create dummy source data
    source_file = "market_data_latest.csv"
    with open(source_file, "w") as f:
        f.write("date,price\n2023-01-01,100\n2023-01-02,101")
    
    print(f"Created source file: {source_file}")
    
    # 2. Create Version (Time Travel Snapshot)
    import time
    version_tag = f"v1.0_{int(time.time())}"
    dataset_name = "test_market_data"
    
    print(f"Creating version {version_tag} for {dataset_name}...")
    dest_path = create_version(source_file, dataset_name, version_tag, metadata={"source": "dummy"})
    print(f"Version created at: {dest_path}")
    
    # 3. Retrieve from Catalog
    print("Retrieving from catalog...")
    dataset = get_dataset(dataset_name, version_tag)
    
    if dataset:
        print(f"Retrieved Dataset: {dataset.name} - {dataset.version}")
        print(f"Path: {dataset.path}")
        assert dataset.path == dest_path
        assert dataset.version == version_tag
    else:
        print("Failed to retrieve dataset!")
        
    # Cleanup
    os.remove(source_file)
    # Optional: cleanup versioned file? Keeping it for inspection usually better.
    print("Test Complete")

if __name__ == "__main__":
    test_data_layer()
