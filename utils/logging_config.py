"""
Logging configuration for EduExtract AI.
"""
import logging
import sys
from pathlib import Path

def get_logger(name: str = "eduextract") -> logging.Logger:
    """Return a configured logger with clean formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
