import logging


def get_logger(name: str) -> logging.Logger:
    """
    Template get a logger
    Args:
        name: name of the logger
    return: Logger
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)
    return logger