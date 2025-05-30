# src/modeling/trainer.py
import joblib
import pandas as pd
import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.utils.config_loader import load_config
from src.utils.data_manager import save_features, save_targets, save_model
from src.utils.logger import get_logger

config = load_config()
logger = get_logger('trainer', config['logs']['log_dir'])

class IncrementalTrainer:
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.le = LabelEncoder()
        self.model_path = self._get_model_path()
        self.load_or_initialize_model()
        
    def _get_model_path(self):
        horizon = config['training']['horizon']
        model_dir = os.path.join(
            config['models']['base_path'], 
            'random_forest', 
            horizon
        )
        return os.path.join(model_dir, "latest_model.pkl")
    
    def load_or_initialize_model(self):
        try:
            self.model, metadata = joblib.load(self.model_path)
            self.feature_columns = metadata.get('feature_columns', [])
            logger.info(f"Loaded existing model from {self.model_path}")
        except (FileNotFoundError, EOFError):
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                warm_start=True,
                random_state=config['training']['random_state'],
                n_jobs=-1
            )
            logger.info("Initialized new Random Forest model")
    
    def prepare_data(self, tech_features, broker_features, targets):
        """Prepare training and test data"""
        # Merge datasets
        features = pd.merge(tech_features, broker_features, on=['Date', 'Symbol'])
        full_data = pd.merge(features, targets, on=['Date', 'Symbol'])
        
        # Filter based on training window
        window = config['training']['training_window']
        if window != 'all' and isinstance(window, int):
            latest_date = full_data['Date'].max()
            cutoff = latest_date - pd.Timedelta(days=window)
            full_data = full_data[full_data['Date'] > cutoff]
        
        # Encode targets
        full_data['Target_Encoded'] = self.le.fit_transform(full_data['Target'])
        
        # Prepare X and y
        X = full_data.drop(columns=['Date', 'Symbol', 'Target', 'Target_Encoded'])
        y = full_data['Target_Encoded']
        self.feature_columns = X.columns.tolist()
        
        # Handle missing values
        X = X.fillna(0)
        
        return train_test_split(
            X, y, 
            test_size=config['training']['test_size'],
            random_state=config['training']['random_state'],
            shuffle=False
        )
    
    def train(self, tech_features, broker_features, targets):
        """Train the model incrementally"""
        X_train, X_test, y_train, y_test = self.prepare_data(tech_features, broker_features, targets)
        
        if X_train is None or X_train.empty:
            logger.error("No training data available")
            return None, None, None
        
        # Incremental training
        start_time = time.time()
        self.model.n_estimators += 10  # Add 10 trees each training
        self.model.fit(X_train, y_train)
        training_duration = time.time() - start_time
        
        # Save updated model
        metadata = {
            'feature_columns': self.feature_columns,
            'label_encoder': self.le,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'horizon': config['training']['horizon'],
            'broker_mode': config['training']['broker_mode']
        }
        
        joblib.dump((self.model, metadata), self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        return X_test, y_test, training_duration