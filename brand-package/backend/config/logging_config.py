"""
Logging Configuration
"""
import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from config.settings import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON
        
        Args:
            record: Log record
            
        Returns:
            JSON formatted string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        
        if hasattr(record, 'generation_id'):
            log_data['generation_id'] = record.generation_id
        
        if hasattr(record, 'project_id'):
            log_data['project_id'] = record.project_id
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add custom fields from extra
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development"""
    
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
    
    # Icons for each level
    ICONS = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🔥'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors
        
        Args:
            record: Log record
            
        Returns:
            Formatted string with colors
        """
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            icon = self.ICONS.get(levelname, '')
            colored_level = (
                f"{self.COLORS[levelname]}{self.BOLD}"
                f"{icon} {levelname:8}"
                f"{self.RESET}"
            )
            record.levelname = colored_level
        
        # Format timestamp
        record.timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        # Format module.function
        record.location = f"{record.module}.{record.funcName}:{record.lineno}"
        
        return super().format(record)


class LoggingConfig:
    """Centralized logging configuration"""
    
    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    CONSOLE_FORMAT_DEV = (
        "%(timestamp)s │ %(levelname)-17s │ "
        "%(name)-25s │ %(message)s"
    )
    
    CONSOLE_FORMAT_PROD = "%(message)s"  # JSON formatted
    
    FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def setup(
        cls,
        log_level: Optional[str] = None,
        log_file: Optional[Path] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: Optional[bool] = None
    ) -> None:
        """Setup logging configuration
        
        Args:
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file
            enable_console: Enable console output
            enable_file: Enable file output
            enable_json: Enable JSON formatting (auto-detected if None)
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
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Auto-detect JSON mode
        if enable_json is None:
            enable_json = settings.app_env == "production"
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Setup console handler
        if enable_console:
            cls._setup_console_handler(root_logger, numeric_level, enable_json)
        
        # Setup file handler
        if enable_file:
            if log_file is None:
                log_file = Path("logs") / f"{settings.app_env}.log"
            cls._setup_file_handler(root_logger, numeric_level, log_file, enable_json)
        
        # Setup specialized loggers
        cls._setup_specialized_loggers(numeric_level)
        
        # Configure third-party loggers
        cls._configure_third_party_loggers(numeric_level)
        
        # Log configuration complete
        logger = logging.getLogger(__name__)
        logger.info(
            f"✅ Logging configured: level={log_level}, "
            f"env={settings.app_env}, json={enable_json}"
        )
    
    @classmethod
    def _setup_console_handler(
        cls,
        logger: logging.Logger,
        level: int,
        use_json: bool
    ) -> None:
        """Setup console handler
        
        Args:
            logger: Logger instance
            level: Log level
            use_json: Use JSON formatting
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if use_json:
            console_handler.setFormatter(JSONFormatter())
        else:
            # Use colored formatter for development
            if settings.app_env == "development":
                formatter = ColoredFormatter(cls.CONSOLE_FORMAT_DEV)
            else:
                formatter = logging.Formatter(cls.DEFAULT_FORMAT)
            console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    @classmethod
    def _setup_file_handler(
        cls,
        logger: logging.Logger,
        level: int,
        log_file: Path,
        use_json: bool
    ) -> None:
        """Setup file handler with rotation
        
        Args:
            logger: Logger instance
            level: Log level
            log_file: Log file path
            use_json: Use JSON formatting
        """
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler (10MB max, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        
        # Set formatter
        if use_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(cls.FILE_FORMAT))
        
        logger.addHandler(file_handler)
    
    @classmethod
    def _setup_specialized_loggers(cls, level: int) -> None:
        """Setup specialized loggers for different components
        
        Args:
            level: Default log level
        """
        # Access logger (for HTTP requests)
        access_logger = logging.getLogger("access")
        access_logger.setLevel(logging.INFO)
        access_logger.propagate = False
        
        if settings.app_env == "production":
            access_file = Path("logs/access.log")
            access_file.parent.mkdir(parents=True, exist_ok=True)
            
            access_handler = logging.handlers.TimedRotatingFileHandler(
                access_file,
                when="midnight",
                interval=1,
                backupCount=30
            )
            access_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(message)s')
            )
            access_logger.addHandler(access_handler)
        
        # Error logger (for critical errors)
        error_logger = logging.getLogger("error")
        error_logger.setLevel(logging.ERROR)
        
        if settings.app_env == "production":
            error_file = Path("logs/error.log")
            error_file.parent.mkdir(parents=True, exist_ok=True)
            
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=10
            )
            error_handler.setFormatter(JSONFormatter())
            error_logger.addHandler(error_handler)
        
        # Security logger (for auth events)
        security_logger = logging.getLogger("security")
        security_logger.setLevel(logging.INFO)
        
        if settings.app_env == "production":
            security_file = Path("logs/security.log")
            security_file.parent.mkdir(parents=True, exist_ok=True)
            
            security_handler = logging.handlers.RotatingFileHandler(
                security_file,
                maxBytes=20 * 1024 * 1024,  # 20MB
                backupCount=20
            )
            security_handler.setFormatter(JSONFormatter())
            security_logger.addHandler(security_handler)
    
    @classmethod
    def _configure_third_party_loggers(cls, level: int) -> None:
        """Configure third-party library loggers
        
        Args:
            level: Default log level
        """
        # Suppress noisy loggers
        noisy_loggers = [
            "urllib3",
            "httpx",
            "httpcore",
            "aiohttp",
            "asyncio",
            "PIL",
            "multipart",
            "uvicorn.access",
            "supabase",
            "openai"
        ]
        
        for logger_name in noisy_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING if level <= logging.INFO else level)
        
        # Configure FastAPI/Uvicorn
        logging.getLogger("uvicorn").setLevel(level)
        logging.getLogger("uvicorn.error").setLevel(level)
        logging.getLogger("fastapi").setLevel(level)
        
        # Configure SQLAlchemy (if used)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    **kwargs
) -> None:
    """Convenience function to setup logging
    
    Args:
        log_level: Log level
        log_file: Log file path
        **kwargs: Additional configuration
    """
    LoggingConfig.setup(
        log_level=log_level,
        log_file=Path(log_file) if log_file else None,
        **kwargs
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger instance
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding context to logs"""
    
    def __init__(self, **kwargs):
        """Initialize with context fields
        
        Args:
            **kwargs: Fields to add to logs
        """
        self.context = kwargs
        self.old_factory = None
    
    def __enter__(self):
        """Enter context"""
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        self.old_factory = logging.getLogRecordFactory()
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        logging.setLogRecordFactory(self.old_factory)


# Utility functions for structured logging
def log_request(logger: logging.Logger, request_data: Dict[str, Any]):
    """Log HTTP request
    
    Args:
        logger: Logger instance
        request_data: Request information
    """
    logger.info(
        "HTTP Request",
        extra={'extra_fields': {
            'request': request_data,
            'type': 'http_request'
        }}
    )


def log_response(logger: logging.Logger, response_data: Dict[str, Any]):
    """Log HTTP response
    
    Args:
        logger: Logger instance
        response_data: Response information
    """
    logger.info(
        "HTTP Response",
        extra={'extra_fields': {
            'response': response_data,
            'type': 'http_response'
        }}
    )


def log_error(logger: logging.Logger, error: Exception, context: Dict[str, Any]):
    """Log error with context
    
    Args:
        logger: Logger instance
        error: Exception
        context: Error context
    """
    logger.error(
        f"Error: {str(error)}",
        exc_info=True,
        extra={'extra_fields': {
            'error_type': type(error).__name__,
            'context': context,
            'type': 'error'
        }}
    )


def log_metric(logger: logging.Logger, metric_name: str, value: float, tags: Dict[str, str]):
    """Log metric for monitoring
    
    Args:
        logger: Logger instance
        metric_name: Metric name
        value: Metric value
        tags: Metric tags
    """
    logger.info(
        f"Metric: {metric_name}",
        extra={'extra_fields': {
            'metric': metric_name,
            'value': value,
            'tags': tags,
            'type': 'metric'
        }}
    )