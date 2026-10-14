import os

from dotenv import load_dotenv


def env_check() -> None:
    """
    Check to see if have a .env file in main project directory
    :return: None
    """
    # Make sure we have a .env file
    if os.path.exists('.env'):
        dotenv_path = '.env'
    else:
        raise FileNotFoundError(
            'No .env or deploy.env file found in current directory')
    load_dotenv(dotenv_path=dotenv_path, override=True)

env_check()