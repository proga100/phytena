import logging
import sys
from typing import Any

def setup_logging():
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Also configure third party loggers to be less chatty if needed
    # logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    
    return logger

logger = setup_logging()
