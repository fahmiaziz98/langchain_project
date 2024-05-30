import asyncio

def async_retry(max_retries: int = 3, delay: float = 1):
    """Async retry decorator with exponential backoff.

    Args:
        max_retries (int): The number of attempts to make before giving up.
        delay (float): The initial delay between attempts. The delay will
            increase exponentially with each attempt.

    Returns:
        A decorator function that can be used to retry an async function.

    Example:
        @async_retry()
        async def my_func():
            # implementation of my_func
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    delay_ = delay * (2 ** (attempt - 1))
                    print(f"Attempt {attempt} failed: {str(e)}")
                    await asyncio.sleep(delay_)
            raise ValueError(f"Failed after {max_retries} attempts")
        return wrapper
    return decorator
