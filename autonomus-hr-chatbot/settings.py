import os
from pathlib import Path
from typing import Union
from dotenv import load_dotenv


def load_env_vars(root_dir: Union[str, Path]) -> dict:
    """
    Load environment variable from .env.default and .env file
    Args:
        root_dir: root directory of the .env files
        returns:
            Dictionary with environment variabel
    """

    if isinstance(root_dir, str):
        root_dir = Path(root_dir)
    load_dotenv(dotenv_path=root_dir / ".env.default")
    load_dotenv(dotenv_path=root_dir / ".env", override=True)

    return dict(os.environ)

def get_root_dir(default_value: str=".") -> Path:
    """
    Get the root directory of the project.

    Args:
        default_value: Default value to use if the environment variable is not set.

    Returns:
        Path to the root directory of the project.
    """

    return Path(os.getenv("API_KEY_ROOT_DIR", default=default_value))


API_KEY_ROOT_DIR = get_root_dir()
SETTINGS = load_env_vars(root_dir=API_KEY_ROOT_DIR)