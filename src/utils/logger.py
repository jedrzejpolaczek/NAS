"""
This module provides a function to create a logger object for logging messages.

Functions:
    get_logger(name, log_file, level=logging.INFO): 
        Creates a logger object with a configured file handler and formatter.
"""
import logging


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
    handler = logging.FileHandler(log_file)
    handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s  %(message)s')
    )
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
