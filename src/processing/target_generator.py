import pandas as pd
import numpy as np
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

config = load_config()
logger = get_logger('target_generator')

def generate_targets(price_data):
    """Create labeled targets for different horizons"""
    if price_data.empty:
        return pd.DataFrame()
    
    thresholds = {
        'next_day': (config['training']['threshold_buy'], 
                     config['training']['threshold_sell']),
        '3day': (0.03, -0.03),
        'weekly': (0.05, -0.05)
    }
    
    horizon = config['training']['horizon']
    buy_thresh, sell_thresh = thresholds[horizon]
    
    # Calculate future returns
    price_data = price_data.sort_values(['Symbol', 'Date'])
    price_data['Future_Price'] = price_data.groupby('Symbol')['Close'].shift(-1)
    
    if horizon == '3day':
        price_data['Future_Price'] = price_data.groupby('Symbol')['Close'].shift(-3)
    elif horizon == 'weekly':
        price_data['Future_Price'] = price_data.groupby('Symbol')['Close'].shift(-5)
    
    price_data['Future_Return'] = (price_data['Future_Price'] / price_data['Close']) - 1
    
    # Create targets
    conditions = [
        price_data['Future_Return'] > buy_thresh,
        price_data['Future_Return'] < sell_thresh
    ]
    choices = ['Buy', 'Sell']
    
    price_data['Target'] = np.select(conditions, choices, default='Hold')
    
    # Drop rows with missing targets
    price_data = price_data.dropna(subset=['Target'])
    
    logger.info(f"Generated targets for {horizon} horizon")
    return price_data[['Date', 'Symbol', 'Target']]