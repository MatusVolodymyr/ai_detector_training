"""
Logging utilities for training pipeline
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def setup_logger(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logger with console and file handlers.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configured logger instance
    """
    log_config = config['logging']
    
    # Create logger
    logger = logging.getLogger('ai_detector_training')
    logger.setLevel(getattr(logging, log_config['level']))
    logger.handlers = []  # Clear any existing handlers
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_config['console']:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_config['file']:
        log_file = log_config['file_path']
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_phase_start(logger: logging.Logger, phase: str, description: str):
    """Log the start of a training phase"""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"{phase}: {description}")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def log_phase_complete(logger: logging.Logger, phase: str, metrics: Dict[str, Any]):
    """Log the completion of a training phase"""
    logger.info("")
    logger.info(f"✓ {phase} completed successfully")
    if metrics:
        logger.info(f"Metrics: {metrics}")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
