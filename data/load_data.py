import aerosandbox as asb
import aerosandbox.numpy as np
import polars as pl
from pathlib import Path

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

df = pl.read_csv(
    Path(__file__).parent / "data.csv",
    dtypes={
        col: pl.Float32
        for col in all_cols
    }
)


def make_airfoil(row_index):
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
    print(df)
    make_airfoil(20).draw()
