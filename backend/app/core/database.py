"""Database configuration with SQLite/PostgreSQL support.

Automatically configures connection based on DATABASE_URL:
- SQLite: Uses NullPool and check_same_thread=False
- PostgreSQL: Uses connection pooling with sensible defaults
"""

from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config.settings import get_settings

# Use database URL from settings (supports env var override)
settings = get_settings()
SQLALCHEMY_DATABASE_URL = settings.database_url


def _create_engine():
    """Create SQLAlchemy engine with appropriate settings for the database type."""
    is_sqlite = SQLALCHEMY_DATABASE_URL.startswith("sqlite")
    
    if is_sqlite:
        # SQLite-specific configuration
        return create_engine(
            SQLALCHEMY_DATABASE_URL,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,  # SQLite doesn't benefit from connection pooling
        )
    else:
        # PostgreSQL/MySQL configuration with connection pooling
        return create_engine(
            SQLALCHEMY_DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections after 1 hour
        )


engine = _create_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for FastAPI endpoints to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables.
    
    Creates all tables defined in the models if they don't exist.
    Safe to call multiple times.
    """
    # Import models to ensure they are registered with Base
    from app.domain import models
    from quant.data import models as quant_models
    Base.metadata.create_all(bind=engine)
