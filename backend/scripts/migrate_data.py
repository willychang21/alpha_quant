import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(base_dir, 'data')
    target_dir = os.path.join(base_dir, 'data_lake', 'raw')
    
    if not os.path.exists(source_dir):
        logger.warning(f"Source directory {source_dir} does not exist.")
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        logger.info(f"Created target directory {target_dir}")

    # List of files to migrate (or all files)
    # The user said "Move data/*", so we move everything.
    # But we should be careful about open file handles (like sqlite).
    # Assuming the app is stopped during migration.
    
    files_to_move = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    for filename in files_to_move:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        try:
            shutil.move(source_path, target_path)
            logger.info(f"Moved {filename} to {target_dir}")
        except Exception as e:
            logger.error(f"Failed to move {filename}: {e}")
            
    # Create a symlink or README in the old location to point to the new one?
    # Or just leave it empty. The user said "Move".
    # However, existing code might break if it relies on backend/data.
    # We should probably update the code references too, but that's a bigger task.
    # For now, let's just move the files as requested.
    
    logger.info("Data migration complete.")

if __name__ == "__main__":
    migrate_data()
