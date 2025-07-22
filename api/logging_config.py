"""Centralized logging configuration for OCR Engine"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any
import uuid
from contextvars import ContextVar

# Context variable for request ID
request_id_var: ContextVar[str] = ContextVar('request_id', default='system')

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'request_id': request_id_var.get(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
            
        # Add custom attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'msecs', 'levelname', 
                          'levelno', 'pathname', 'filename', 'module', 'exc_info',
                          'exc_text', 'stack_info', 'lineno', 'funcName', 'getMessage']:
                log_obj[key] = value
                
        return json.dumps(log_obj)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for development"""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    blue = "\x1b[34;20m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: green + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def setup_logging(app_name: str = "ocr-engine", log_level: str = None) -> None:
    """
    Set up application logging with appropriate formatters and handlers
    
    Args:
        app_name: Name of the application for logging
        log_level: Override log level (defaults to env var or INFO)
    """
    # Determine environment and log level
    env = os.getenv('ENVIRONMENT', 'development')
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'DEBUG' if env == 'development' else 'INFO')
    
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Use JSON formatter in production, colored in development
    if env == 'production':
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter())
    
    root_logger.addHandler(console_handler)
    
    # Add file handler for errors
    if env == 'production':
        error_file_handler = logging.handlers.RotatingFileHandler(
            '/tmp/ocr-engine-errors.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(error_file_handler)
    
    # Set specific log levels for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Log initial setup
    logger = logging.getLogger(app_name)
    logger.info(
        "Logging initialized",
        extra={
            'environment': env,
            'log_level': log_level,
            'handlers': len(root_logger.handlers),
            'pid': os.getpid()
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


def set_request_id(request_id: str = None) -> str:
    """Set request ID for current context"""
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id


def log_performance(logger: logging.Logger, operation: str, duration_ms: float, 
                   success: bool = True, **extra) -> None:
    """Log performance metrics"""
    logger.info(
        f"Performance: {operation}",
        extra={
            'operation': operation,
            'duration_ms': round(duration_ms, 2),
            'success': success,
            **extra
        }
    )


def log_memory_usage(logger: logging.Logger, context: str) -> Dict[str, Any]:
    """Log current memory usage"""
    import psutil
    import torch
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    memory_data = {
        'context': context,
        'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
        'vms_mb': round(memory_info.vms / 1024 / 1024, 2),
        'percent': round(process.memory_percent(), 2),
        'available_mb': round(psutil.virtual_memory().available / 1024 / 1024, 2)
    }
    
    # Add GPU memory if available
    if torch.cuda.is_available():
        gpu_memory = {
            'gpu_allocated_mb': round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
            'gpu_reserved_mb': round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
            'gpu_max_memory_mb': round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
        }
        memory_data.update(gpu_memory)
    
    logger.info(f"Memory usage: {context}", extra=memory_data)
    return memory_data