import os
import glob
import pandas as pd
import sys

# === 1. Load all cleaned floor sheet data ===
clean_files = sorted(glob.glob("data/processed/clean_sheet_data_*.csv"))
if not clean_files:
    print("❌ No clean_sheet_data_*.csv files found in data/processed/")
    sys.exit(1)

clean_dfs = [pd.read_csv(file) for file in clean_files]
clean_data = pd.concat(clean_dfs, ignore_index=True)

# === 2. Load all technical/broker indicators ===
indicator_files = sorted(glob.glob("data/indicators/indicators_*.csv"))
if not indicator_files:
    print("❌ No indicators_*.csv files found in data/indicators/")
    sys.exit(1)

indicator_dfs = [pd.read_csv(file) for file in indicator_files]
indicator_data = pd.concat(indicator_dfs, ignore_index=True)

# === 3. Merge datasets ===
full_data = pd.merge(
    clean_data,
    indicator_data,
    left_on=["Symbol", "ContractNo", "Buyer", "Seller"],
    right_on=["Symbol", "ContractNo_x", "Buyer_x", "Seller_x"],
    how="inner"
)

# === 4. Fix column conflicts ===
full_data["Date"] = pd.to_datetime(full_data["Date_x"])
full_data["Rate"] = full_data["Rate_x"]
full_data["Amount"] = full_data["Amount_x"]

# Drop unnecessary columns
drop_cols = [col for col in full_data.columns if col.endswith("_x") or col.endswith("_y")]
drop_cols = [col for col in drop_cols if col not in ["Rate_x", "Amount_x", "Date_x"]]
full_data = full_data.drop(columns=drop_cols)

# === 5. Sort + Target (Tomorrow's Rate) ===
full_data = full_data.sort_values(by=["Symbol", "Date"])
full_data["TomorrowRate"] = full_data.groupby("Symbol")["Rate"].shift(-1)

# Drop rows with missing target
full_data = full_data.dropna(subset=["TomorrowRate"])

# === 6. Save to CSV ===
os.makedirs("data/training_data", exist_ok=True)
full_data.to_csv("data/training_data/train_dataset.csv", index=False)

print("✅ Dataset created at: data/training_data/train_dataset.csv")
