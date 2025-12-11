"""Database Migrations Runner.

Applies pending Alembic migrations on application startup.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_migrations():
    """Apply pending database migrations.
    
    Runs Alembic 'upgrade head' to apply all pending migrations.
    Called during application startup before init_db().
    
    Raises:
        Exception: If migrations fail (prevents app startup)
    """
    try:
        from alembic.config import Config
        from alembic import command
        
        # Find alembic.ini relative to backend directory
        backend_dir = Path(__file__).parent.parent.parent
        alembic_ini = backend_dir / "alembic.ini"
        
        if not alembic_ini.exists():
            logger.warning(
                "[MIGRATIONS] alembic.ini not found, skipping migrations. "
                "Run 'alembic init alembic' to set up migrations."
            )
            return
        
        alembic_cfg = Config(str(alembic_ini))
        
        # Set script location relative to alembic.ini
        alembic_cfg.set_main_option("script_location", str(backend_dir / "alembic"))
        
        command.upgrade(alembic_cfg, "head")
        logger.info("[MIGRATIONS] Database migrations applied successfully")
        
    except ImportError:
        logger.warning(
            "[MIGRATIONS] Alembic not installed, skipping migrations. "
            "Install with: pip install alembic"
        )
    except Exception as e:
        logger.error(f"[MIGRATIONS] Failed to apply migrations: {e}")
        raise
