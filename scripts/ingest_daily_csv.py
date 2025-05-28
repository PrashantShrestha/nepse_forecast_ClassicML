import pandas as pd
from pathlib import Path

# Define directories
raw_dir = Path("data/raw")
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

print("📁 Source directory:", raw_dir)

# List to hold all cleaned DataFrames for combining later
all_cleaned_dfs = []

# Process each raw file
for raw_file in sorted(raw_dir.glob("floor_sheet_data_*.csv")):
    date_part = raw_file.stem.replace("floor_sheet_data_", "")
    clean_file = processed_dir / f"clean_sheet_data_{date_part}.csv"

    # Skip if already cleaned
    if clean_file.exists():
        print(f"⏩ Skipping '{raw_file.name}' (already processed)")
        # Load existing cleaned file to add to combined later
        df_existing = pd.read_csv(clean_file)
        all_cleaned_dfs.append(df_existing)
        continue

    print(f"⏬ Cleaning '{raw_file.name}'...")

    try:
        # Read CSV
        df = pd.read_csv(raw_file, dtype=str)

        # Standardize column names
        df.columns = df.columns.str.strip()

        # Clean numeric columns
        for col in ['Quantity', 'Rate', 'Amount']:
            df[col] = df[col].str.replace(",", "", regex=False).astype(float)

        # Convert types
        df['SN'] = df['SN'].astype(int)
        df['ContractNo'] = df['ContractNo'].astype(str)
        df['Buyer'] = df['Buyer'].astype(int)
        df['Seller'] = df['Seller'].astype(int)

        # Add date column
        df['Date'] = pd.to_datetime(date_part).date()

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Drop rows with missing values
        df.dropna(inplace=True)

        # Save to cleaned directory
        df.to_csv(clean_file, index=False)
        print(f"✅ Saved: {clean_file.name}")

        # Append cleaned DataFrame to list for combining later
        all_cleaned_dfs.append(df)

    except Exception as e:
        print(f"❌ Error processing '{raw_file.name}': {e}")

