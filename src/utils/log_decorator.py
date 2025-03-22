import logging
import time
from functools import wraps
from typing import Any, Callable
import inspect

# Configure logging to write to debug_log.txt
logging.basicConfig(
    filename='debug_log.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'  # Append mode to store all logs in one file
)
logger = logging.getLogger(__name__)

def log_function_call(func: Callable) -> Callable:
    """
    A decorator that logs detailed information about function calls.

    Logs:
    - When the function is called (timestamp via logging).
    - The calling object and its class type (if called from a class).
    - Entry parameter names, values, and types.
    - Return value and its type.
    - Execution time in seconds.

    Args:
        func (Callable): The function to wrap.

    Returns:
        Callable: The wrapped function.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Start timing
        start_time = time.time()

        # Determine the calling context (self if from a class)
        caller = "Unknown"
        caller_type = "N/A"
        if args and inspect.isclass(getattr(args[0], "__class__", None)):
            caller = args[0]  # Instance (self)
            caller_type = type(caller).__name__

        # Prepare parameter details
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        param_details = [
            f"{name}={repr(value)} (type: {type(value).__name__})"
            for name, value in bound_args.arguments.items()
        ]

        # Log function entry
        entry_msg = (
            f"Function '{func.__name__}' called by {caller} (type: {caller_type})\n"
            f"Parameters: {', '.join(param_details)}"
        )
        logger.debug(entry_msg)

        # Execute the function
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            # Log exception and re-raise
            duration = time.time() - start_time
            error_msg = (
                f"Function '{func.__name__}' raised {type(e).__name__}: {str(e)}\n"
                f"Execution time: {duration:.4f} seconds"
            )
            logger.debug(error_msg)
            raise

        # Calculate execution time
        duration = time.time() - start_time

        # Log return value and execution time
        return_msg = (
            f"Function '{func.__name__}' returned: {repr(result)} (type: {type(result).__name__})\n"
            f"Execution time: {duration:.4f} seconds"
        )
        logger.debug(return_msg)

        return result

    return wrapper
