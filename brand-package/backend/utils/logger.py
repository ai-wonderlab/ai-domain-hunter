"""
Centralized Logging Configuration
"""
import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import traceback

from config.settings import settings


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m'   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        # Add color to log level
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{self.BOLD}"
                f"{levelname}{self.RESET}"
            )
        
        # Format timestamp
        record.timestamp = datetime.fromtimestamp(
            record.created
        ).strftime('%H:%M:%S')
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        
        # Add exception info
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data)


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None
) -> None:
    """
    Configure application logging
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
    """
    # Determine log level
    if log_level is None:
        if settings.debug:
            log_level = "DEBUG"
        elif settings.app_env == "production":
            log_level = "INFO"
        else:
            log_level = "DEBUG"
    
    # Convert to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console Handler (with colors)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if settings.app_env == "development":
        # Use colored formatter in development
        console_format = (
            "%(timestamp)s │ %(levelname)-8s │ %(name)-20s │ %(message)s"
        )
        console_handler.setFormatter(ColoredFormatter(console_format))
    else:
        # Use JSON formatter in production
        console_handler.setFormatter(JSONFormatter())
    
    root_logger.addHandler(console_handler)
    
    # File Handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler (10MB max, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        
        # Always use JSON for file logging
        file_handler.setFormatter(JSONFormatter())
        
        root_logger.addHandler(file_handler)
    
    # Access log (separate file)
    access_logger = logging.getLogger("access")
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False
    
    if settings.app_env == "production":
        access_file = Path("logs/access.log")
        access_file.parent.mkdir(parents=True, exist_ok=True)
        
        access_handler = logging.handlers.RotatingFileHandler(
            access_file,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        access_handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        access_logger.addHandler(access_handler)
    
    # Configure third-party loggers
    configure_third_party_loggers(numeric_level)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"✅ Logging configured (level: {log_level})")
    logger.info(f"   Environment: {settings.app_env}")
    logger.info(f"   Log file: {log_file or 'None (console only)'}")


def configure_third_party_loggers(level: int) -> None:
    """Configure third-party library loggers"""
    
    # Suppress noisy loggers
    noisy_loggers = [
        "urllib3",
        "httpx",
        "httpcore",
        "openai",
        "asyncio",
        "multipart.multipart",
        "PIL"
    ]
    
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING if level <= logging.INFO else level)
    
    # FastAPI/Uvicorn loggers
    logging.getLogger("uvicorn").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding context to logs"""
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self.old_factory = None
    
    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        self.old_factory = logging.getLogRecordFactory()
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)