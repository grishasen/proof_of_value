import asyncio
import time
from functools import wraps
from typing import Callable

from value_dashboard.utils.logger import get_logger

logger = get_logger(__name__)


def timed(func: Callable) -> Callable:
    """Decorator that logs the execution time of a sync or async function in milliseconds."""

    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            result = await func(*args, **kwargs)
            end = time.time()
            elapsed_ms = (end - start) * 1000
            logger.debug(f"{func.__name__} executed in {elapsed_ms:.2f} ms (async)")
            return result

        return async_wrapper

    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            elapsed_ms = (end - start) * 1000
            logger.debug(f"{func.__name__} executed in {elapsed_ms:.2f} ms")
            return result

        return sync_wrapper
