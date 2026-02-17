import yaml
import os

def load_config(config_path="config.yaml"):
    """
    Loads the YAML configuration file from the project root.
    """
    # Get absolute path to ensure we find the file regardless of execution context
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_path, config_path)

    try:
        with open(full_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {full_path}")
        # Default fallback
        return {"rewards": {"invalid_action": -10, "win": 50, "loss": -50}}