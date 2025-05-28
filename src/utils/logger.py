import logging
import os
import pandas as pd
import json
from datetime import datetime
from src.utils.config_loader import load_config
# src/utils/logger.py
import logging
import os
from src.utils.config_loader import load_config

config = load_config()

def setup_logger(name, log_file, level=logging.INFO):
    """Create and configure logger"""
    log_path = os.path.join(config['logs']['log_dir'], log_file)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_path)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

def log_evaluation(date, metrics, horizon):
    """Log evaluation results to CSV and JSON"""
    # Add metadata to metrics
    metrics['date'] = date
    metrics['horizon'] = horizon
    metrics['timestamp'] = datetime.now().isoformat()
    
    # CSV logging
    csv_path = os.path.join(config['logs']['log_dir'], "evaluation_history.csv")
    try:
        history = pd.read_csv(csv_path)
        history = pd.concat([history, pd.DataFrame([metrics])], ignore_index=True)
    except FileNotFoundError:
        history = pd.DataFrame([metrics])
    
    history.to_csv(csv_path, index=False)
    
    # JSON logging
    json_path = os.path.join(config['logs']['log_dir'], f"evaluation_{date}.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return history


# config = load_config()

# def setup_logger(name, log_file, level=logging.INFO):
#     """Create and configure logger"""
#     log_path = os.path.join(config['logs']['log_dir'], log_file)
    
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
#     handler = logging.FileHandler(log_path)
#     handler.setFormatter(formatter)
    
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     logger.addHandler(handler)
    
#     return logger

# def log_evaluation(date, metrics, horizon):
#     """Log evaluation results to CSV and JSON"""
#     # Add metadata to metrics
#     metrics['date'] = date
#     metrics['horizon'] = horizon
#     metrics['timestamp'] = datetime.now().isoformat()
    
#     # CSV logging
#     csv_path = os.path.join(config['logs']['log_dir'], "evaluation_history.csv")
#     try:
#         history = pd.read_csv(csv_path)
#         history = pd.concat([history, pd.DataFrame([metrics])], ignore_index=True)
#     except FileNotFoundError:
#         history = pd.DataFrame([metrics])
    
#     history.to_csv(csv_path, index=False)
    
#     # JSON logging
#     json_path = os.path.join(config['logs']['log_dir'], f"evaluation_{date}.json")
#     with open(json_path, 'w') as f:
#         json.dump(metrics, f, indent=2)
    
#     return history