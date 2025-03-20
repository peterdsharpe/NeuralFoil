import csv
import polars as pl
from pathlib import Path

# Get the current directory
data_directory = Path(__file__).parent

# Find all CSV files in the directory
csv_files = list(data_directory.glob("data*.csv"))

if not csv_files:
    raise ValueError("No CSV files found in the current directory.")

print(f"Found {len(csv_files)} CSV files to process.")

# Read and concatenate all CSV files
dfs = []
for csv_file in csv_files:
    print(f"Reading {csv_file}...")
    df = pl.read_csv(csv_file, has_header=False)
    print(f"\tRead {len(df)} rows")
    dfs.append(df)

# Concatenate all dataframes
combined_df = pl.concat(dfs)
print(f"Combined dataframe has {len(combined_df)} rows")

# Save the combined dataframe to a new CSV file
output_file = data_directory / "data_xfoil.csv"
combined_df.write_csv(output_file, include_header=False)
print(f"Saved combined data to {output_file}")
