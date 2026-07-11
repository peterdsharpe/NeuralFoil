import aerosandbox.numpy as np
import polars as pl
from pathlib import Path
from neuralfoil._basic_data_type import Data

cols = Data.get_vector_column_names()

### Read the original data, by scraping all .csv files within the data directory
data_directory = Path(__file__).parent

raw_dfs = {}

for csv_file in data_directory.glob("data*.csv"):
    print(f"Reading {csv_file}...")
    raw_dfs[csv_file.stem] = pl.read_csv(
        csv_file, has_header=False, schema_overrides={col: pl.Float32 for col in cols}
    )
    print(f"\t{len(raw_dfs[csv_file.stem])} rows")

df = pl.concat(raw_dfs.values())

# Do some basic cleanup
cols_to_nullify = Data.get_vector_output_column_names().copy()
cols_to_nullify.remove("analysis_confidence")

c = pl.col("CD") <= 0
print(f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with CD <= 0...")
df = df.with_columns(
    [
        pl.when(c)
        .then(0)
        .otherwise(pl.col("analysis_confidence"))
        .alias("analysis_confidence"),
    ]
    + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal(
    [pl.col(f"upper_bl_theta_{i}") <= 0 for i in range(Data.N)]
    + [pl.col(f"lower_bl_theta_{i}") <= 0 for i in range(Data.N)]
)
print(
    f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with nonpositive boundary layer thetas..."
)
df = df.with_columns(
    [
        pl.when(c)
        .then(0)
        .otherwise(pl.col("analysis_confidence"))
        .alias("analysis_confidence"),
    ]
    + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal(
    [pl.col(f"upper_bl_H_{i}") < 1 for i in range(Data.N)]
    + [pl.col(f"lower_bl_H_{i}") < 1 for i in range(Data.N)]
)
print(
    f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with H < 1 (non-physical BL)..."
)
df = df.with_columns(
    [
        pl.when(c)
        .then(0)
        .otherwise(pl.col("analysis_confidence"))
        .alias("analysis_confidence"),
    ]
    + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal(
    sum(
        [
            [
                pl.col(f"upper_bl_ue/vinf_{i}") < -20,
                pl.col(f"upper_bl_ue/vinf_{i}") > 20,
                pl.col(f"lower_bl_ue/vinf_{i}") < -20,
                pl.col(f"lower_bl_ue/vinf_{i}") > 20,
            ]
            for i in range(Data.N)
        ],
        start=[],
    )
)
print(
    f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with non-physical edge velocities..."
)
df = df.with_columns(
    [
        pl.when(c)
        .then(0)
        .otherwise(pl.col("analysis_confidence"))
        .alias("analysis_confidence"),
    ]
    + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal(
    pl.col("Top_Xtr") < 0,
    pl.col("Top_Xtr") > 1,
    pl.col("Bot_Xtr") < 0,
    pl.col("Bot_Xtr") > 1,
)
print(
    f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with non-physical transition locations..."
)
df = df.with_columns(
    [
        pl.when(c)
        .then(0)
        .otherwise(pl.col("analysis_confidence"))
        .alias("analysis_confidence"),
    ]
    + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

print("Dataset:")
print(df)
# print("Dataset statistics:")
# print(df.describe())

### Shuffle the training set (deterministically)
df = df.sample(fraction=1, with_replacement=False, shuffle=True, seed=0)

# Make the scaled datasets. The scaling transformations are shared with
# inference (see neuralfoil/_core.py and neuralfoil/_spec.py), so training
# targets and inference decoding cannot drift apart.
from .scaling import scale_inputs, scale_outputs

df_inputs_scaled = scale_inputs(df)
di = df_inputs_scaled.describe()

df_outputs_scaled = scale_outputs(df)
do = df_outputs_scaled.describe([0.01, 0.99])

### Split the dataset into train and test sets
test_train_split_index = int(len(df) * 0.95)
# df_train = df[:test_train_split_index]
# df_test = df[test_train_split_index:]
df_train_inputs_scaled = df_inputs_scaled[:test_train_split_index]
df_train_outputs_scaled = df_outputs_scaled[:test_train_split_index]
df_test_inputs_scaled = df_inputs_scaled[test_train_split_index:]
df_test_outputs_scaled = df_outputs_scaled[test_train_split_index:]

mean_inputs_scaled = np.mean(df_inputs_scaled.to_numpy(), axis=0)
cov_inputs_scaled = np.cov(df_inputs_scaled.to_numpy(), rowvar=False)


def make_data(row_index, df=df):
    row = df[row_index]
    return Data.from_vector(row[cols].to_numpy().flatten())


if __name__ == "__main__":
    d = make_data(len(df) // 2)
