import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from src.utils.config_loader import load_config
from src.utils.data_manager import load_features
from src.utils.logger import get_logger

config = load_config()
logger = get_logger('predictor')

class Predictor:
    def __init__(self, horizon=None):
        self.horizon = horizon or config['training']['horizon']
        self.model, self.metadata = self.load_latest_model()
        self.le = self.metadata.get('label_encoder')
        self.feature_columns = self.metadata.get('feature_columns', [])
        self.features = self.load_features()
    
    def load_latest_model(self):
        """Load the most recent model for the current horizon"""
        model_dir = os.path.join(
            config['models']['base_path'], 
            'random_forest', 
            self.horizon
        )
        model_path = os.path.join(model_dir, "latest_model.pkl")
        
        try:
            model, metadata = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model, metadata
        except (FileNotFoundError, EOFError) as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
    def load_features(self):
        """Load required features"""
        tech_features = load_features('technical')
        broker_features = load_features(f"broker_{config['training']['broker_mode']}")
        
        # Merge features
        if tech_features.empty or broker_features.empty:
            return pd.DataFrame()
            
        features = pd.merge(tech_features, broker_features, on=['Date', 'Symbol'])
        return features.dropna()
    
    def get_current_features(self, symbol):
        """Get latest features for a specific symbol"""
        if self.features.empty:
            return None
            
        latest_date = self.features['Date'].max()
        symbol_features = self.features[
            (self.features['Symbol'] == symbol) & 
            (self.features['Date'] == latest_date)
        ]
        
        if symbol_features.empty:
            logger.warning(f"No features found for {symbol} on {latest_date}")
            return None
        
        # Prepare feature vector
        feature_vector = symbol_features.drop(columns=['Date', 'Symbol'])
        
        # Align columns with training features
        missing_cols = set(self.feature_columns) - set(feature_vector.columns)
        extra_cols = set(feature_vector.columns) - set(self.feature_columns)
        
        for col in missing_cols:
            feature_vector[col] = 0
            
        return feature_vector[self.feature_columns]
    
    def predict(self, symbol):
        """Generate prediction for a symbol"""
        features = self.get_current_features(symbol)
        if features is None:
            return {
                'symbol': symbol,
                'signal': 'Unavailable',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
        
        # Make prediction
        prediction_encoded = self.model.predict(features)[0]
        prediction = self.le.inverse_transform([prediction_encoded])[0]
        
        probabilities = self.model.predict_proba(features)[0]
        confidence = probabilities.max()
        
        # Get class probabilities
        class_probs = {
            cls: prob for cls, prob in zip(self.le.classes_, probabilities)
        }
        
        return {
            'symbol': symbol,
            'signal': prediction,
            'confidence': confidence,
            'probabilities': class_probs,
            'features': features.iloc[0].to_dict(),
            'timestamp': datetime.now().isoformat(),
            'model_version': self.metadata.get('training_date', 'unknown')
        }
    
    def batch_predict(self, symbols):
        """Predict for multiple symbols"""
        return {symbol: self.predict(symbol) for symbol in symbols}