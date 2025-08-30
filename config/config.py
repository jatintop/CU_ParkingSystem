import toml
import os

class Config:
    def __init__(self, toml_path="pyproject.toml"):
        self.config_data = self._load_config(toml_path)
        self.parking_system = self.config_data.get('tool', {}).get('parking_system', {})

    def _load_config(self, toml_path):
        if not os.path.exists(toml_path):
            raise FileNotFoundError(f"Configuration file not found at {toml_path}")
        with open(toml_path, 'r') as f:
            return toml.load(f)

    def get(self, key, default=None):
        return self.parking_system.get(key, default)

# Initialize the configuration
settings = Config()
