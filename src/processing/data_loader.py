# src/processing/data_loader.py
import pandas as pd
import os
import numpy as np
from glob import glob
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

config = load_config()
logger = get_logger('data_loader', config['logs']['log_dir'])

def load_raw_data():
    """Load and concatenate all raw CSV files"""
    raw_files = glob(os.path.join(config['data']['raw_path'], 'floor_sheet_data_*.csv'))
    
    if not raw_files:
        logger.warning("No raw data files found")
        return pd.DataFrame()
    
    dfs = []
    
    for file in raw_files:
        try:
            # Extract date from filename
            date_str = os.path.basename(file).split('_')[-1].replace('.csv', '')
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(date_str)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")
    
    if not dfs:
        return pd.DataFrame()
    
    full_df = pd.concat(dfs).sort_values('Date')
    
    # Data cleaning
    full_df = full_df.dropna(subset=['Symbol', 'Rate', 'Quantity'])
 # Clean Quantity column - remove commas and convert to int
    def clean_quantity(value):
        if isinstance(value, str):
            return int(value.replace(',', ''))
        return int(value)
    
    full_df['Quantity'] = full_df['Quantity'].apply(clean_quantity)
    
    # Clean Rate column
    full_df['Rate'] = full_df['Rate'].astype(float)
    
    # Handle amount formatting
    if 'Amount' in full_df.columns:
        full_df['Amount'] = full_df['Amount'].str.replace(',', '').astype(float)
    
    # Rename columns for clarity
    column_map = {
        'Buyer': 'BuyerBroker',
        'Seller': 'SellerBroker'
    }
    full_df = full_df.rename(columns={k: v for k, v in column_map.items() if k in full_df.columns})
    
    # Create buy/sell flags
    full_df['Trade_Type'] = np.where(full_df['BuyerBroker'].notna(), 'Buy', 'Sell')
    
    logger.info(f"Loaded {len(full_df)} records from {len(raw_files)} files")
    return full_df