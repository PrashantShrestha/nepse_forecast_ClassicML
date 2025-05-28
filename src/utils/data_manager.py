import pandas as pd
import os
import joblib
import json
from src.utils.config_loader import load_config

config = load_config()

def save_features(df, feature_name):
    """Save processed features to disk"""
    path = os.path.join(config['data']['features_path'], f"{feature_name}.feather")
    df.reset_index(drop=True).to_feather(path)

def load_features(feature_name):
    """Load processed features from disk"""
    path = os.path.join(config['data']['features_path'], f"{feature_name}.feather")
    return pd.read_feather(path)

def save_targets(df):
    """Save targets to disk"""
    path = os.path.join(config['data']['targets_path'], "targets.feather")
    df.to_feather(path)

def load_targets():
    """Load targets from disk"""
    path = os.path.join(config['data']['targets_path'], "targets.feather")
    return pd.read_feather(path)

def save_model(model, path, metadata):
    """Save model with metadata"""
    joblib.dump(model, path)
    with open(path.replace('.pkl', '.json'), 'w') as f:
        json.dump(metadata, f)

def load_model(path):
    """Load model with metadata"""
    model = joblib.load(path)
    with open(path.replace('.pkl', '.json'), 'r') as f:
        metadata = json.load(f)
    return model, metadata