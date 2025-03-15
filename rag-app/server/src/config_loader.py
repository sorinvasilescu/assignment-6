# TODO: confirm I can remove this.
import os
import yaml


class ConfigLoader:
    _config = None

    @classmethod
    def load_config(cls, config_name: str):
        """
        Loads the configuration file from the config/ directory.
        This method caches the configuration to avoid reloading multiple times.
        """
        if cls._config is None:
            # Determine the base path of the configuration directory
            config_path = os.path.join(
                os.path.dirname(__file__), "../config", f"{config_name}.yaml"
            )

            # Load the YAML config file
            with open(config_path, "r") as file:
                cls._config = yaml.safe_load(file)

        return cls._config

    @classmethod
    def get_config_value(cls, key: str, default=None):
        """
        Retrieve a specific config value from the loaded configuration.
        """
        if cls._config is None:
            raise ValueError("Configuration not loaded. Please call load_config first.")

        return cls._config.get(key, default)
