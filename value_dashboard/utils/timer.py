import logging
import time
from functools import wraps

from value_dashboard.utils.logger import get_logger

logger = get_logger(__name__, logging.DEBUG)


def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("{} ran in {:.2f}ms".format(func.__name__, (end - start) * 10 ** 3))
        return result

    return wrapper
