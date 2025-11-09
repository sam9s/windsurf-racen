import os
from pathlib import Path
from dotenv import load_dotenv


def load_env() -> None:
    """
    Load environment variables from a .env file in the project root if present.
    """
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    # Also honor existing environment without overriding
