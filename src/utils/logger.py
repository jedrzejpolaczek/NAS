"""
This module provides a function to create a logger object for logging messages.

Functions:
    get_logger(name, log_file, level=logging.INFO): 
        Creates a logger object with a configured file handler and formatter.
"""
import os
from logging.handlers import RotatingFileHandler
import logging

# Configuration constants
DEFAULT_LOG_DIR = "logs"
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5



def get_logger(
    name: str,
    log_file: str,
    level=logging.INFO
) -> logging.Logger:
    """
    Creates a logger object with a configured file handler and formatter.

    Args:
        name (str):
            The name of the logger.
        log_file (str):
            The path to the file where logs will be written.
        level (int, optional):
            The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger:
            The created logger object.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Ensure log directory exists and validate path
    log_dir = os.path.abspath(DEFAULT_LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.abspath(
        os.path.join(
            log_dir,
            os.path.basename(log_file)
        )
    )
    if not log_path.startswith(log_dir):
        raise ValueError("Log file must be within the configured log directory")

    try:
        handler = RotatingFileHandler(
            log_path,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT
        )
        logger.setLevel(level)
        logger.addHandler(handler)

    except (IOError, PermissionError) as e:
        raise RuntimeError(
            f"Failed to initialize logger: {e}"
        ) from e

    # Prevent propagation to root logger
    logger.propagate = False

    return logger
