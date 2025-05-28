import yaml
import os

def load_config():
    # Get the base directory of the project
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create necessary directories
    os.makedirs(config['data']['raw_path'], exist_ok=True)
    os.makedirs(config['data']['processed_path'], exist_ok=True)
    os.makedirs(config['data']['features_path'], exist_ok=True)
    os.makedirs(config['data']['targets_path'], exist_ok=True)
    os.makedirs(config['logs']['log_dir'], exist_ok=True)
    
    # Model directories
    model_base = os.path.join(config['models']['base_path'], 'random_forest')
    for horizon in ['next_day', '3day', 'weekly']:
        os.makedirs(os.path.join(model_base, horizon), exist_ok=True)
    
    return config
