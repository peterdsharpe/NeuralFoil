import aerosandbox as asb
import aerosandbox.numpy as np
import polars as pl
from pathlib import Path
import sys, os

sys.path.append(str(Path(__file__).parent))

from _basic_high_dim_data_type import Data

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
    print(f"\t{len(raw_dfs[csv_file.stem])} rows")

df = pl.concat(raw_dfs.values())

# Do some basic cleanup
cols_to_nullify = Data.get_vector_output_column_names().copy()
cols_to_nullify.remove("analysis_confidence")

c = pl.col("CD") <= 0
print(
    f"Eliminating {int(df.select(c).sum().to_numpy()[0, 0])} rows with CD <= 0..."
)
df = df.with_columns(
    [
        pl.when(c).then(0).otherwise(pl.col("analysis_confidence")).alias("analysis_confidence"),
    ] + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal([
                          pl.col(f"upper_bl_theta_{i}") <= 0
                          for i in range(Data.N)
                      ] + [
                          pl.col(f"lower_bl_theta_{i}") <= 0
                          for i in range(Data.N)
                      ])
print(
    f"Eliminating {int(df.select(c).sum().to_numpy()[0, 0])} rows with nonpositive boundary layer thetas..."
)
df = df.with_columns(
    [
        pl.when(c).then(0).otherwise(pl.col("analysis_confidence")).alias("analysis_confidence"),
    ] + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal([
                          pl.col(f"upper_bl_H_{i}") < 1
                          for i in range(Data.N)
                      ] + [
                          pl.col(f"lower_bl_H_{i}") < 1
                          for i in range(Data.N)
                      ])
print(
    f"Eliminating {int(df.select(c).sum().to_numpy()[0, 0])} rows with H < 1 (non-physical BL)..."
)
df = df.with_columns(
    [
        pl.when(c).then(0).otherwise(pl.col("analysis_confidence")).alias("analysis_confidence"),
    ] + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal(sum([
    [
        pl.col(f"upper_bl_ue/vinf_{i}") < -20,
        pl.col(f"upper_bl_ue/vinf_{i}") > 20,
        pl.col(f"lower_bl_ue/vinf_{i}") < -20,
        pl.col(f"lower_bl_ue/vinf_{i}") > 20,
    ]
    for i in range(Data.N)
], start=[])
)
print(
    f"Eliminating {int(df.select(c).sum().to_numpy()[0, 0])} rows with non-physical edge velocities..."
)
df = df.with_columns(
    [
        pl.when(c).then(0).otherwise(pl.col("analysis_confidence")).alias("analysis_confidence"),
    ] + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

print("Dataset:")
print(df)
print("Dataset statistics:")
print(df.describe())

### Shuffle the training set
df = df.sample(
    fraction=1,
    with_replacement=False,
    shuffle=True,
    seed=0
)

# Make the scaled datasets
df_inputs_scaled = pl.DataFrame({
    **{
        f"s_kulfan_upper_{i}": df[f"kulfan_upper_{i}"]
        for i in range(8)
    },
    **{
        f"s_kulfan_lower_{i}": df[f"kulfan_lower_{i}"]
        for i in range(8)
    },
    "s_kulfan_LE_weight"   : df["kulfan_LE_weight"],
    "s_kulfan_TE_thickness": df["kulfan_TE_thickness"] * 50,
    "s_sin_2a"             : np.sind(2 * df["alpha"]),
    "s_cos_a"              : np.cosd(df["alpha"]),
    "s_cos2_a"             : 1 - np.cosd(df["alpha"]) ** 2,
    "s_Re"                 : (np.log(df["Re"]) - 12.5) / 3.5,
    # No mach
    "s_n_crit"             : (df["n_crit"] - 9) / 4.5,
    "s_xtr_upper"          : df["xtr_upper"],
    "s_xtr_lower"          : df["xtr_lower"],
})

di = df_inputs_scaled.describe()

sqrt_Rex_approx = [((Data.bl_x_points[i] + 1e-2) / df["Re"]) ** 0.5 for i in range(Data.N)]

df_outputs_scaled = pl.DataFrame({
    "s_analysis_confidence": df["analysis_confidence"],
    "s_CL"                 : 2 * df["CL"],
    "s_ln_CD"              : np.log(df["CD"]) / 2 + 2,
    "s_CM"                 : 20 * df["CM"],
    "s_Top_Xtr"            : df["Top_Xtr"],
    "s_Bot_Xtr"            : df["Bot_Xtr"],
    **{
        f"s_upper_bl_ret_{i}": np.log10(np.abs(df[f"upper_bl_ue/vinf_{i}"]) * df[f"upper_bl_theta_{i}"] * df["Re"] + 0.1)
        for i in range(Data.N)
    },
    **{
        f"s_upper_bl_H_{i}": np.log(df[f"upper_bl_H_{i}"] / 2.6)
        for i in range(Data.N)
    },
    **{
        f"s_upper_bl_ue/vinf_{i}": df[f"upper_bl_ue/vinf_{i}"]
        for i in range(Data.N)
    },
    **{
        f"s_lower_bl_ret_{i}": np.log10(np.abs(df[f"lower_bl_ue/vinf_{i}"]) * df[f"lower_bl_theta_{i}"] * df["Re"] + 0.1)
        for i in range(Data.N)
    },
    **{
        f"s_lower_bl_H_{i}": np.log(df[f"lower_bl_H_{i}"] / 2.6)
        for i in range(Data.N)
    },
    **{
        f"s_lower_bl_ue/vinf_{i}": df[f"lower_bl_ue/vinf_{i}"]
        for i in range(Data.N)
    },
})

do = df_outputs_scaled.describe([0.01, 0.99])

### Split the dataset into train and test sets
test_train_split_index = int(len(df) * 0.95)
# df_train = df[:test_train_split_index]
# df_test = df[test_train_split_index:]
df_train_inputs_scaled = df_inputs_scaled[:test_train_split_index]
df_train_outputs_scaled = df_outputs_scaled[:test_train_split_index]
df_test_inputs_scaled = df_inputs_scaled[test_train_split_index:]
df_test_outputs_scaled = df_outputs_scaled[test_train_split_index:]


def make_data(row_index, df=df):
    row = df[row_index]
    return Data.from_vector(
        row[cols].to_numpy().flatten()
    )


if __name__ == '__main__':
    d = make_data(len(df) // 2)
