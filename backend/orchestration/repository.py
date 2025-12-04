from dagster import repository, asset, define_asset_job, ScheduleDefinition, load_assets_from_modules
import logging
from app.core.logging_config import setup_logging
from backend.orchestration import assets

# Ensure logging is configured
setup_logging()
logger = logging.getLogger("dagster")

@repository
def quant_repository():
    return load_assets_from_modules([assets])
