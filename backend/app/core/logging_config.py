import logging
import logging.handlers
import os

def setup_logging():
    """
    Configure centralized logging with rotation.
    Logs are saved to backend/logs/app.log with rotation (5MB, 3 backups).
    """
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'app.log')
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=3
    )
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Log file: {log_file}")
