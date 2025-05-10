import configparser
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent / "config.ini"
    parser = configparser.ConfigParser()
    parser.read(config_path)
    return parser

