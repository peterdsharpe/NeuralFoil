"""
Tests of the shared encode/decode transforms in neuralfoil._core:

- encode_outputs and decode_outputs must be exact inverses (this is what
  guarantees training targets and inference decoding cannot drift apart);
- the polars pathway used by the training data pipeline
  (training/training_data/scaling.py) must produce the same numbers as the
  NumPy pathway used at inference.
"""

import numpy as np
import pytest

from neuralfoil import _core, _spec
from neuralfoil._basic_data_type import Data


def _random_user_outputs(rng, n_cases):
    """Random-but-physical user-facing outputs, keyed like a training DataFrame."""
    outputs = {
        "analysis_confidence": rng.uniform(0, 1, n_cases),
        "CL": rng.uniform(-2, 3, n_cases),
        "CD": rng.uniform(1e-3, 0.5, n_cases),
        "CM": rng.uniform(-0.4, 0.2, n_cases),
        "Top_Xtr": rng.uniform(0, 1, n_cases),
        "Bot_Xtr": rng.uniform(0, 1, n_cases),
    }
    for side in ["upper", "lower"]:
        for i in range(Data.N):
            outputs[f"{side}_bl_theta_{i}"] = rng.uniform(1e-6, 1e-2, n_cases)
            outputs[f"{side}_bl_H_{i}"] = rng.uniform(1.0, 20.0, n_cases)
            outputs[f"{side}_bl_ue/vinf_{i}"] = rng.uniform(-2, 2, n_cases)
    return outputs


def test_encode_decode_outputs_roundtrip():
    rng = np.random.default_rng(0)
    n_cases = 13
    Re = 10 ** rng.uniform(4, 8, n_cases)
    outputs = _random_user_outputs(rng, n_cases)

    latent = _core.encode_outputs(outputs, Re=Re)
    y_fused = np.stack([np.asarray(latent[name]) for name in _spec.OUTPUT_NAMES], axis=1)

    # The confidence column of decode passes through a sigmoid (it is a logit
    # at that point in inference); invert it for the round-trip comparison.
    decoded = _core.decode_outputs(y_fused, Re=Re)
    decoded["analysis_confidence"] = _core.sigmoid(y_fused[:, 0])  # identity here

    for key, expected in outputs.items():
        if key == "analysis_confidence":
            continue  # not a round-trip quantity (label vs. sigmoid-of-logit)
        np.testing.assert_allclose(
            decoded[key], expected, rtol=1e-9, atol=1e-12, err_msg=key
        )


def test_polars_scaling_matches_numpy_encode():
    """
    The training-side polars pathway must produce the same latent values as
    the inference-side NumPy pathway, from the same raw data.
    """
    pl = pytest.importorskip("polars")
    scaling = pytest.importorskip(
        "training.training_data.scaling",
        reason="training code not present (e.g., running from sdist)",
    )

    rng = np.random.default_rng(0)
    n_cases = 13

    ### Build a synthetic raw-training-data DataFrame
    raw = {
        **{f"kulfan_upper_{i}": rng.uniform(0, 0.4, n_cases) for i in range(8)},
        **{f"kulfan_lower_{i}": rng.uniform(-0.4, 0.1, n_cases) for i in range(8)},
        "kulfan_LE_weight": rng.uniform(-0.3, 0.3, n_cases),
        "kulfan_TE_thickness": rng.uniform(0, 0.01, n_cases),
        "alpha": rng.uniform(-20, 20, n_cases),
        "Re": 10 ** rng.uniform(4, 8, n_cases),
        "n_crit": rng.uniform(0, 18, n_cases),
        "xtr_upper": rng.uniform(0, 1, n_cases),
        "xtr_lower": rng.uniform(0, 1, n_cases),
        **_random_user_outputs(rng, n_cases),
    }
    df = pl.DataFrame(raw)

    ### Inputs: polars pathway vs. direct NumPy encode
    df_inputs_scaled = scaling.scale_inputs(df)
    assert df_inputs_scaled.columns == [f"s_{name}" for name in _spec.INPUT_NAMES]

    numpy_rows = _core.encode_inputs(
        kulfan_parameters={
            "upper_weights": [raw[f"kulfan_upper_{i}"] for i in range(8)],
            "lower_weights": [raw[f"kulfan_lower_{i}"] for i in range(8)],
            "leading_edge_weight": raw["kulfan_LE_weight"],
            "TE_thickness": raw["kulfan_TE_thickness"],
        },
        alpha=raw["alpha"],
        Re=raw["Re"],
        n_crit=raw["n_crit"],
        xtr_upper=raw["xtr_upper"],
        xtr_lower=raw["xtr_lower"],
    )
    for name, numpy_row in zip(_spec.INPUT_NAMES, numpy_rows):
        np.testing.assert_allclose(
            df_inputs_scaled[f"s_{name}"].to_numpy(), numpy_row, rtol=1e-12, err_msg=name
        )

    ### Outputs: polars pathway vs. direct NumPy encode
    df_outputs_scaled = scaling.scale_outputs(df)
    assert df_outputs_scaled.columns == [f"s_{name}" for name in _spec.OUTPUT_NAMES]

    numpy_latent = _core.encode_outputs(
        outputs={key: raw[key] for key in raw}, Re=raw["Re"]
    )
    for name in _spec.OUTPUT_NAMES:
        np.testing.assert_allclose(
            df_outputs_scaled[f"s_{name}"].to_numpy(),
            numpy_latent[name],
            rtol=1e-12,
            err_msg=name,
        )


if __name__ == "__main__":
    pytest.main([__file__])
