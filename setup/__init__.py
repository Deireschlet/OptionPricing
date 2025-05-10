# loads the setup.ini file
from .config_loader import load_config
from .logger import get_logger


config = load_config()

logger = get_logger(config["PROJECT"]["name"])
