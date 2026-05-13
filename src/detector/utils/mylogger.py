import logging.config
import json
from src.detector.config import Config
from pathlib import Path

def setup_logging():
    config_file = Path(Config.LOG_CONFIG_PATH)
    
    with open(config_file) as f_in:
        config = json.load(f_in)
        
    logging.config.dictConfig(config)