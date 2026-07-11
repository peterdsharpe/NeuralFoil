"""
Builds the scaled (latent-space) training DataFrames from raw XFoil data.

The actual transformations live in neuralfoil._core (encode_inputs /
encode_outputs) and neuralfoil._spec, shared with inference - this module
only adapts them to polars DataFrames. Kept free of import-time side effects
so it is testable without training data present.
"""

import polars as pl

from neuralfoil import _core, _spec


def scale_inputs(df: pl.DataFrame) -> pl.DataFrame:
    """Raw training data -> scaled network-input DataFrame (25 columns, `s_` prefix)."""
    rows = _core.encode_inputs(
        kulfan_parameters={
            "upper_weights": [df[f"kulfan_upper_{i}"] for i in range(8)],
            "lower_weights": [df[f"kulfan_lower_{i}"] for i in range(8)],
            "leading_edge_weight": df["kulfan_LE_weight"],
            "TE_thickness": df["kulfan_TE_thickness"],
        },
        alpha=df["alpha"],
        Re=df["Re"],
        n_crit=df["n_crit"],
        xtr_upper=df["xtr_upper"],
        xtr_lower=df["xtr_lower"],
    )
    return pl.DataFrame(
        {f"s_{name}": row for name, row in zip(_spec.INPUT_NAMES, rows)}
    )


def scale_outputs(df: pl.DataFrame) -> pl.DataFrame:
    """Raw training data -> scaled network-target DataFrame (198 columns, `s_` prefix)."""
    latent = _core.encode_outputs(outputs=df, Re=df["Re"])
    return pl.DataFrame({f"s_{name}": latent[name] for name in _spec.OUTPUT_NAMES})
