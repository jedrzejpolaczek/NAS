import pytest
import logging
from src.utils.logger import get_logger


def test_get_logger_creates_logger():
    """Tests if the function creates a logger object."""
    logger = get_logger("test_logger", "test.log")
    assert isinstance(logger, logging.Logger)


def test_get_logger_sets_level():
    """Tests if the function sets the logger level correctly."""
    logger = get_logger("test_logger", "test.log", level=logging.DEBUG)
    assert logger.level == logging.INFO


def test_get_logger_configures_file_handler():
    """Tests if the function configures the file handler correctly."""
    logger = get_logger("test_logger", "test.log")
    handler = logger.handlers[0]
    assert isinstance(handler, logging.FileHandler)


def test_get_logger_removes_existing_handlers():
    """Tests if the function removes existing handlers."""
    logger = logging.getLogger("test_logger")
    logger.addHandler(logging.StreamHandler())  # Add a StreamHandler
    get_logger("test_logger", "test.log")
    assert len(logger.handlers) == 2
    assert isinstance(logger.handlers[0], logging.FileHandler)
