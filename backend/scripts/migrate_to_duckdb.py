import duckdb
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_to_duckdb():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sqlite_path = os.path.join(base_dir, 'data_lake', 'raw', 'database.sqlite')
    duckdb_path = os.path.join(base_dir, 'warehouse', 'catalog.duckdb')
    
    if not os.path.exists(sqlite_path):
        logger.error(f"SQLite database not found at {sqlite_path}")
        return

    logger.info(f"Migrating from {sqlite_path} to {duckdb_path}...")
    
    con = duckdb.connect(duckdb_path)
    
    try:
        # Install sqlite extension
        con.execute("INSTALL sqlite;")
        con.execute("LOAD sqlite;")
        
        # Attach SQLite
        con.execute(f"ATTACH '{sqlite_path}' AS sqlite_db (TYPE SQLITE);")
        
        # Get tables
        tables = con.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        
        for table in tables:
            table_name = table[0]
            logger.info(f"Migrating table: {table_name}")
            
            # Create table in DuckDB and copy data
            con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM sqlite_db.{table_name};")
            
        logger.info("Migration to DuckDB complete.")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    migrate_to_duckdb()
