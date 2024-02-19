import aerosandbox as asb
import aerosandbox.numpy as np
import polars as pl
from pathlib import Path
from data_types import Data

cols = Data.get_vector_column_names()

### Read the original data, by scraping all .csv files within the data directory
data_directory = Path(__file__).parent

raw_dfs = {}

for csv_file in data_directory.glob("data*.csv"):
    print(f"Reading {csv_file}...")
    raw_dfs[csv_file.stem] = pl.read_csv(
        csv_file, has_header=False,
        dtypes={
            col: pl.Float32
            for col in cols
        }
    )

df = pl.concat(raw_dfs.values())

print("Dataset:")
print(df)
print("Dataset statistics:")
print(df.describe())

### Split the dataset into train and test sets
test_train_split_index = int(len(df) * 0.95)
df_train = df[:test_train_split_index]
df_test = df[test_train_split_index:]

### Shuffle the training set
df_train = df_train.sample(
    fraction=1,
    with_replacement=False,
    shuffle=True,
    seed=0
)


def make_data(row_index, df=df):
    row = df[row_index]
    return Data.from_vector(
        row[cols].to_numpy().flatten()
    )


if __name__ == '__main__':
    make_data(0).airfoil.draw()
