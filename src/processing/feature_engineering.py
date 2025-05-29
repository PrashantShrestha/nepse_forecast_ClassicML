import pandas as pd
import numpy as np
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

config = load_config()
logger = get_logger('feature_engineering')

def calculate_technical_features(df):
    """Compute technical indicators for each symbol"""
    if df.empty:
        return pd.DataFrame()
    
    # Calculate daily summary
    daily_summary = df.groupby(['Symbol', 'Date']).agg(
        Open=('Rate', 'first'),
        High=('Rate', 'max'),
        Low=('Rate', 'min'),
        Close=('Rate', 'last'),
        Volume=('Quantity', 'sum')
    ).reset_index()
    
    # Sort by date
    daily_summary = daily_summary.sort_values(['Symbol', 'Date'])
    
    # Technical indicators
    def compute_ma(group, window=5):
        return group.rolling(window, min_periods=1).mean()
    
    def compute_std(group, window=14):
        return group.rolling(window, min_periods=1).std()
    
    def compute_rsi(group, window=14):
        delta = group.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window, min_periods=1).mean()
        avg_loss = loss.rolling(window, min_periods=1).mean()
        
        # Handle division by zero
        with np.errstate(divide='ignore'):
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        rsi.replace([np.inf, -np.inf], 100 if avg_loss == 0 else 0, inplace=True)
        return rsi
    
    # Calculate features
    daily_summary['5d_ma'] = daily_summary.groupby('Symbol')['Close'].transform(compute_ma)
    daily_summary['14d_std'] = daily_summary.groupby('Symbol')['Close'].transform(compute_std)
    daily_summary['14d_rsi'] = daily_summary.groupby('Symbol')['Close'].transform(compute_rsi)
    
    # Volatility
    daily_summary['Volatility'] = daily_summary['14d_std'] / daily_summary['5d_ma']
    
    # Daily price change
    daily_summary['Daily_Return'] = daily_summary.groupby('Symbol')['Close'].pct_change()
    
    logger.info("Calculated technical features")
    return daily_summary[['Date', 'Symbol', '5d_ma', '14d_std', '14d_rsi', 'Volatility', 'Daily_Return', 'Volume']]

def calculate_broker_features(df, mode='relative'):
    """Compute broker behavior metrics"""
    if df.empty:
        return pd.DataFrame()
    
    # Create broker activity DataFrame
    broker_activity = pd.DataFrame()
    
    # Buyer activity
    buyer_activity = df[df['Trade_Type'] == 'Buy'].groupby(
        ['Date', 'Symbol', 'BuyerBroker']
    ).agg(Buy_Volume=('Quantity', 'sum')).reset_index()
    
    # Seller activity
    seller_activity = df[df['Trade_Type'] == 'Sell'].groupby(
        ['Date', 'Symbol', 'SellerBroker']
    ).agg(Sell_Volume=('Quantity', 'sum')).reset_index()
    
    # Merge buyer and seller activity
    broker_activity = pd.merge(
        buyer_activity.rename(columns={'BuyerBroker': 'Broker'}),
        seller_activity.rename(columns={'SellerBroker': 'Broker'}),
        on=['Date', 'Symbol', 'Broker'],
        how='outer'
    ).fillna(0)
    
    # Net strength calculation
    if mode == 'relative':
        broker_activity['Net_Strength'] = (
            (broker_activity['Buy_Volume'] - broker_activity['Sell_Volume']) / (broker_activity['Buy_Volume'] + broker_activity['Sell_Volume'] + 1e-6))
    else:
        broker_activity['Net_Strength'] = broker_activity['Buy_Volume'] - broker_activity['Sell_Volume']
    
    # Broker concentration (HHI)
    symbol_daily = broker_activity.groupby(['Date', 'Symbol']).apply(
        lambda x: (x['Net_Strength'].abs() / x['Net_Strength'].abs().sum()).pow(2).sum()
    ).reset_index(name='Broker_HHI')
    
    # Large block trades (> 1 million NPR)
    df['Trade_Value'] = df['Quantity'] * df['Rate']
    large_trades = df[df['Trade_Value'] > 1e6].groupby(
        ['Date', 'Symbol']
    ).size().reset_index(name='Large_Trades_Count')
    
    # Merge features
    broker_features = pd.merge(symbol_daily, large_trades, on=['Date', 'Symbol'], how='left')
    broker_features['Large_Trades_Count'] = broker_features['Large_Trades_Count'].fillna(0)
    
    logger.info(f"Calculated broker features ({mode} mode)")
    return broker_features