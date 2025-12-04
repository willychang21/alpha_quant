from quant.model_registry.schema import ModelMetadata, ModelConfig
from quant.model_registry.snapshots import create_snapshot, get_snapshot
import os
import json

def test_registry():
    # 1. Create dummy data
    data_path = "dummy_data.csv"
    with open(data_path, "w") as f:
        f.write("col1,col2\n1,2\n3,4")
        
    print(f"Created dummy data at {data_path}")
    
    # 2. Create Snapshot
    file_hash = create_snapshot(data_path, "Test Snapshot")
    print(f"Created Snapshot Hash: {file_hash}")
    
    # 3. Verify Snapshot Retrieval
    snap = get_snapshot(file_hash)
    print(f"Retrieved Snapshot Metadata: {json.dumps(snap, indent=2)}")
    assert snap['hash'] == file_hash
    
    # 4. Create Model Metadata with new fields
    metadata = ModelMetadata(
        model_id="test_model_v2",
        version="2.0.0",
        type="valuation",
        config=ModelConfig(parameters={"wacc": 0.08}),
        training_data_hash=file_hash,
        git_commit_hash="abc1234"
    )
    
    print(f"Model Metadata Validated:\n{metadata.model_dump_json(indent=2)}")
    
    # Cleanup
    os.remove(data_path)
    print("Test Complete")

if __name__ == "__main__":
    test_registry()
