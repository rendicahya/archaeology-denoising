import importlib.util
from pathlib import Path


def load_config(config_path):
    config_path = Path(config_path)
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    config.config_name = config_path.stem

    spec.loader.exec_module(config)

    return config
