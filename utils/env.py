import os
from typing import Optional

def load_env(dotenv_path: Optional[str] = None) -> None:
    """Load environment variables from a .env file if present.

    Looks for `.env` in the project root by default.
    Sets common keys like OPENAI_API_KEY, HUGGINGFACE_HUB_TOKEN, WANDB_API_KEY.
    """
    try:
        # Import lazily to avoid hard dependency issues
        from dotenv import load_dotenv
    except Exception:
        # If python-dotenv isn't installed, skip silently. Caller may still
        # provide env vars through the shell environment.
        return

    # Determine default .env path if not provided
    if dotenv_path is None:
        dotenv_path = os.path.join(os.getcwd(), ".env")

    # Load variables if .env exists
    if os.path.isfile(dotenv_path):
        load_dotenv(dotenv_path)


