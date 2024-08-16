import logging
import time
from functools import wraps

logger = logging.getLogger("AutoInspection")


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        if hasattr(args[0], '__class__'):  # Check if the first argument has a class (for methods)
            class_name = args[0].__class__.__name__
            logger.info(
                f"{'Method' if class_name else 'Function'} '{class_name}.{func.__name__}' took {end_time - start_time:.6f} seconds to execute.")
        else:
            logger.info(f"Function '{func.__name__}' took {end_time - start_time:.6f} seconds to execute.")
        return result

    return wrapper
