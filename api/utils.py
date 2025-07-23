"""Utility functions for the API"""

import asyncio
import functools
import signal
import time
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from logging_config import get_logger

logger = get_logger(__name__)


class TimeoutException(Exception):
    """Custom timeout exception"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Operation timed out")


def timeout(seconds: int):
    """
    Decorator to add timeout to synchronous functions
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set up the signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            except TimeoutException:
                logger.error(f"{func.__name__} timed out after {seconds} seconds")
                raise
            finally:
                # Restore the old handler and cancel the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        
        return wrapper
    return decorator


def async_timeout(seconds: int):
    """
    Decorator to add timeout to async functions
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"{func.__name__} timed out after {seconds} seconds")
                raise TimeoutException(f"Operation timed out after {seconds} seconds")
        
        return wrapper
    return decorator


def run_with_timeout(func: Callable, args: tuple = (), kwargs: dict = None, 
                    timeout_seconds: int = 60) -> Any:
    """
    Run a function with a timeout using ThreadPoolExecutor
    
    Args:
        func: Function to run
        args: Positional arguments
        kwargs: Keyword arguments
        timeout_seconds: Timeout in seconds
        
    Returns:
        Function result
        
    Raises:
        TimeoutException: If function times out
    """
    if kwargs is None:
        kwargs = {}
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except FutureTimeoutError:
            logger.error(f"{func.__name__} timed out after {timeout_seconds} seconds")
            # Try to cancel the future
            future.cancel()
            raise TimeoutException(f"Operation timed out after {timeout_seconds} seconds")
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise


def chunked_processing(items: list, chunk_size: int = 10) -> list:
    """
    Process items in chunks to avoid memory issues
    
    Args:
        items: List of items to process
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of items
    """
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]