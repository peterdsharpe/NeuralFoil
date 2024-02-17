import aerosandbox as asb
import aerosandbox.numpy as np
import polars as pl
from pathlib import Path

### Describe the data we're expecting, in terms of what columns (cols) we want to see.
aero_input_cols = [
    'alpha',
    'Re',
]

aero_output_cols = [
    'CL',
    'CD',
    'CM',
    'Cpmin',
    'Top_Xtr',
    'Bot_Xtr',
]

kulfan_lower_cols = [
    f"kulfan_lower_{i}" for i in range(8)
]

kulfan_upper_cols = [
    f"kulfan_upper_{i}" for i in range(8)
]

kulfan_cols = kulfan_lower_cols + kulfan_upper_cols + [
    "kulfan_TE_thickness",
    "kulfan_LE_weight",
]

all_cols = aero_input_cols + aero_output_cols + kulfan_cols


### Read the original data, by scraping all .csv files within the data directory
data_directory = Path(__file__).parent

dfs_list = []

for  csv_file in data_directory.glob("*.csv"):
    print(f"Reading {csv_file}...")
    dfs_list.append(pl.read_csv(
        csv_file,
        dtypes={
            col: pl.Float32
            for col in all_cols
        }
    ))

df = pl.concat(dfs_list)

### Double the dataset by adding a flipped version of each airfoil
# No longer necessary, since we've implemented a symmetry constraint on the network at train time

# # Create a copy of the original dataframe
# df_flipped = df_original.clone()
#
# # Modify the columns in the copy
# df_flipped = df_flipped.with_columns(
#     pl.col('alpha') * -1,
#     pl.col('CL') * -1,
#     pl.col('CM') * -1,
#     pl.col('kulfan_LE_weight') * -1,
# )
# # df_flipped = df_flipped.with_columns(pl.col('kulfan_LE_weight') * -1)
#
# original_top_xtr = df_flipped['Top_Xtr'].clone()
# original_bot_xtr = df_flipped['Bot_Xtr'].clone()
#
# df_flipped.replace('Top_Xtr', original_bot_xtr)
# df_flipped.replace('Bot_Xtr', original_top_xtr)
#
# for i in range(8):
#     lower_column = f"kulfan_lower_{i}"
#     upper_column = f"kulfan_upper_{i}"
#
#     # Store the original columns
#     original_lower = df_flipped[lower_column].clone()
#     original_upper = df_flipped[upper_column].clone()
#
#     # Swap and negate the columns
#     df_flipped.replace(lower_column, original_upper * -1)
#     df_flipped.replace(upper_column, original_lower * -1)
#
# # Combine the original and the modified dataframes
# df = pl.concat([df_original, df_flipped])

### Drop all rows with CD <= 0
df = df.filter(pl.col('CD') > 0)

### Shuffle the dataset, and compute some basic statistics
df = df.sample(
    fraction=1,
    with_replacement=False,
    shuffle=True,
    seed=0
)

print("Dataset:")
print(df)
print("Dataset statistics:")
print(df.describe())

### Split the dataset into train and test sets
test_train_split_index = int(len(df) * 0.95)
df_train = df[:test_train_split_index]
df_test = df[test_train_split_index:]


def get_weight(row: pl.Series):
    alpha_center = 4
    alpha_width = 8
    return np.exp(-((row["alpha"] - alpha_center) / alpha_width) ** 2)


weights = get_weight(df_train)


def make_airfoil(row_index, df=df):
    row = df[row_index]
    kulfan_params = dict(
        lower_weights=row[kulfan_lower_cols].to_numpy().flatten(),
        upper_weights=row[kulfan_upper_cols].to_numpy().flatten(),
        TE_thickness=float(row["kulfan_TE_thickness"].to_numpy()),
        LE_weight=float(row["kulfan_LE_weight"].to_numpy()),
    )
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
    coordinates = get_kulfan_coordinates(**kulfan_params)

    return asb.Airfoil(coordinates=coordinates)


if __name__ == '__main__':
    make_airfoil(-20).draw()
