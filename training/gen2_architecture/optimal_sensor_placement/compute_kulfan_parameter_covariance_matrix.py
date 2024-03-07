from load_data import df, Data
import aerosandbox as asb
import numpy as np

kulfans_data = np.stack([
    df[f"kulfan_{side}_{i}"]
    for side in [
        "upper",
        "lower"
    ]
    for i in range(8)
] + [
    df["kulfan_LE_weight"],
    df["kulfan_TE_thickness"]
], axis=1)

cov_data = np.cov(kulfans_data, rowvar=False)

airfoil_database_path = asb._asb_root / "geometry" / "airfoil" / "airfoil_database"

airfoil_database = [
    asb.Airfoil(name=filename.stem).normalize().to_kulfan_airfoil()
    for filename in airfoil_database_path.glob("*.dat")
]

kulfans_database = np.stack([
    np.concatenate([
        airfoil.upper_weights,
        airfoil.lower_weights,
        np.atleast_1d(airfoil.leading_edge_weight),
        np.atleast_1d(airfoil.TE_thickness)
    ])
    for airfoil in airfoil_database
], axis=0)

cov_database = np.cov(kulfans_database, rowvar=False)