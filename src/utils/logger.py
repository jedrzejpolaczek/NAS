import logging


def get_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
