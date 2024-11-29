import logging
import sys
from pathlib import Path
from typing import Optional, Any
from logging.handlers import RotatingFileHandler


class LogConfigError(Exception):
    """Base exception for logging configuration errors."""
    pass


class TerminalFormatter(logging.Formatter):
    """Custom formatter for terminal output with simplified format.
    
    Formats log messages differently based on their level, omitting timestamps
    and other metadata for cleaner terminal output.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record based on its level.
        
        Args:
            record: The log record to format.
            
        Returns:
            str: The formatted log message.
        """
        if record.levelno == logging.INFO:
            return f"{record.getMessage()}"
        elif record.levelno == logging.WARNING:
            return f"Warning: {record.getMessage()}"
        elif record.levelno == logging.ERROR:
            return f"Error: {record.getMessage()}"
        elif record.levelno == logging.DEBUG:
            return f"Debug: {record.getMessage()}"
        return super().format(record)


def setup_logger(
    name: str, 
    log_dir: Optional[Path] = Path("logs"),
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Configure and return a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_dir: Directory for log files. If None, file logging is disabled
        console_level: Logging level for console output
        file_level: Logging level for file output
        max_bytes: Maximum size of each log file
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
        
    Raises:
        LogConfigError: If logger configuration fails
    """
    try:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Clear any existing handlers   
        logger.handlers.clear()

        # Terminal handler with simplified format
        terminal_handler = logging.StreamHandler(sys.stdout)
        terminal_handler.setLevel(console_level)
        terminal_handler.setFormatter(TerminalFormatter())
        logger.addHandler(terminal_handler)

        # File handler with detailed format (if log_dir provided)
        if log_dir:
            _log_config_add_file_handler(
                log_dir, name, logger, 
                file_level, max_bytes, backup_count
            )
        return logger
        
    except Exception as e:
        raise LogConfigError(f"Failed to setup logger: {str(e)}") from e


def _log_config_add_file_handler(
    log_dir: Path,
    name: str, 
    logger: logging.Logger,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10_485_760,
    backup_count: int = 5
) -> None:
    """Add a rotating file handler to the logger.
    
    Args:
        log_dir: Directory for log files
        name: Logger name
        logger: Logger instance to configure
        file_level: Logging level for file output
        max_bytes: Maximum size of each log file
        backup_count: Number of backup files to keep
        
    Raises:
        LogConfigError: If file handler configuration fails
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / f"{name.lower()}.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S.%f",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
    except Exception as e:
        raise LogConfigError(f"Failed to setup file handler: {str(e)}") from e
