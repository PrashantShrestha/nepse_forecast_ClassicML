import pandas as pd
from pathlib import Path

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Paths
input_dir = Path("data/processed")  # Contains clean_sheet_data_*.csv
output_dir = Path("data/indicators")
output_dir.mkdir(parents=True, exist_ok=True)

# Get all daily clean data files, sorted
input_files = sorted(input_dir.glob("clean_sheet_data_2025-*.csv"))

for input_file in input_files:
    try:
        date_str = input_file.stem.split("_")[-1]
        output_file = output_dir / f"indicators_{date_str}.csv"

        if output_file.exists():
            print(f"⏭️ Skipped: {output_file.name} (already exists)")
            continue

        df = pd.read_csv(input_file)

        # Calculate indicators on this single file only
        df['TypicalPrice'] = df['Rate']
        df['VWAP'] = df['Amount'] / df['Quantity']
        df['MA_5'] = df.groupby('Symbol')['Rate'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        df['RSI'] = df.groupby('Symbol')['Rate'].transform(lambda x: calculate_rsi(x, period=14))
        df['Volatility'] = df.groupby('Symbol')['Rate'].transform(lambda x: x.rolling(window=5, min_periods=1).std().fillna(0))

        # Prepare columns for merge structure (as before)
        df_main = df.rename(columns={
            'SN': 'SN_x', 'ContractNo': 'ContractNo_x', 'Buyer': 'Buyer_x',
            'Seller': 'Seller_x', 'Quantity': 'Quantity_x', 'Rate': 'Rate_x',
            'Amount': 'Amount_x'
        })

        df_merge = df_main.rename(columns={
            'SN_x': 'SN_y', 'ContractNo_x': 'ContractNo_y', 'Buyer_x': 'Buyer_y',
            'Seller_x': 'Seller_y', 'Quantity_x': 'Quantity_y', 'Rate_x': 'Rate_y',
            'Amount_x': 'Amount_y'
        })

        df_merge['BuyerStrength'] = df_merge['Amount_y'] * df_merge['Buyer_y']
        df_merge['SellerStrength'] = df_merge['Amount_y'] * df_merge['Seller_y']
        df_merge['NetBuy'] = df_merge['BuyerStrength'] - df_merge['SellerStrength']

        final_cols = [
            'SN_x', 'ContractNo_x', 'Symbol', 'Buyer_x', 'Seller_x', 'Quantity_x',
            'Rate_x', 'Amount_x', 'Date', 'MA_5', 'RSI', 'VWAP', 'Volatility',
            'SN_y', 'ContractNo_y', 'Buyer_y', 'Seller_y', 'Quantity_y', 'Rate_y',
            'Amount_y', 'BuyerStrength', 'SellerStrength', 'NetBuy'
        ]

        # Merge on index to get final_df columns as expected
        final_df = pd.merge(df_main, df_merge[[
            'SN_y', 'ContractNo_y', 'Buyer_y', 'Seller_y', 'Quantity_y', 'Rate_y',
            'Amount_y', 'BuyerStrength', 'SellerStrength', 'NetBuy'
        ]], left_index=True, right_index=True)

        final_df = final_df[final_cols]

        # Save indicators file per day, no combined file
        final_df.to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file.name}")

    except Exception as e:
        print(f"❌ Failed: {input_file.name} — {e}")
