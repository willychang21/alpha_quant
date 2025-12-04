from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Use the existing database from the server directory
# Assuming the backend is run from the root or backend directory, we need to locate it.
# We'll use an absolute path or relative to the project root.
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'database.sqlite')
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
print(f"DEBUG: Using Database at {DB_PATH}")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=NullPool
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # Import models to ensure they are registered with Base
    from app.domain import models
    from quant.data import models as quant_models
    Base.metadata.create_all(bind=engine)
