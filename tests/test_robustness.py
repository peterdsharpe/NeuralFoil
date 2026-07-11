"""
Tests of NeuralFoil's robustness guarantees: every shipped model loads and
runs, an answer is always returned (even far outside the training
distribution), and malformed inputs raise clear errors.
"""

import numpy as np
import pytest

import neuralfoil as nf

ALL_MODEL_SIZES = [
    "xxsmall",
    "xsmall",
    "small",
    "medium",
    "large",
    "xlarge",
    "xxlarge",
    "xxxlarge",
]

KULFAN = dict(
    upper_weights=np.array([0.20, 0.25, 0.20, 0.27, 0.20, 0.25, 0.20, 0.20]),
    lower_weights=np.array([-0.10, -0.05, -0.10, 0.00, 0.05, 0.05, 0.10, 0.10]),
    leading_edge_weight=0.15,
    TE_thickness=0.002,
)


@pytest.mark.parametrize("model_size", ALL_MODEL_SIZES)
def test_all_model_sizes_load_and_run(model_size):
    """Every shipped weights file must load and produce finite, plausible outputs."""
    aero = nf.get_aero_from_kulfan_parameters(
        KULFAN, alpha=5.0, Re=1e6, model_size=model_size
    )
    for key in aero:
        assert np.all(np.isfinite(aero[key])), key
    assert 0 < float(aero["CD"][0]) < 1
    assert -1 < float(aero["CL"][0]) < 3


@pytest.mark.parametrize("alpha", [-180.0, -90.0, 90.0, 180.0])
@pytest.mark.parametrize("Re", [1e2, 1e10])
def test_always_returns_an_answer(alpha, Re):
    """
    Far outside the training distribution, NeuralFoil must still return finite
    values (with low confidence), never NaN/Inf - this is the "guaranteed to
    return an answer" promise that distinguishes it from XFoil.
    """
    aero = nf.get_aero_from_kulfan_parameters(
        KULFAN, alpha=alpha, Re=Re, model_size="large"
    )
    for key in ["analysis_confidence", "CL", "CD", "CM", "Top_Xtr", "Bot_Xtr"]:
        assert np.all(np.isfinite(aero[key])), key
    assert 0 <= float(aero["analysis_confidence"][0]) <= 1
    assert float(aero["CD"][0]) > 0


def test_invalid_model_size_raises():
    with pytest.raises(ValueError, match="model_size"):
        nf.get_aero_from_kulfan_parameters(
            KULFAN, alpha=5.0, Re=1e6, model_size="gigantic"
        )


def test_mismatched_vector_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        nf.get_aero_from_kulfan_parameters(
            KULFAN, alpha=np.linspace(0, 5, 3), Re=np.full(4, 1e6)
        )


@pytest.mark.parametrize("n_weights", [7, 12])
def test_wrong_kulfan_weight_count_raises(n_weights):
    bad_kulfan = dict(
        upper_weights=np.full(n_weights, 0.2),
        lower_weights=np.full(n_weights, -0.1),
        leading_edge_weight=0.15,
        TE_thickness=0.002,
    )
    with pytest.raises(ValueError, match="8 CST weights per side"):
        nf.get_aero_from_kulfan_parameters(bad_kulfan, alpha=5.0, Re=1e6)


if __name__ == "__main__":
    pytest.main([__file__])
