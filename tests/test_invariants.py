"""
Tests of the physics invariants that NeuralFoil's architecture structurally
guarantees, plus the basic output contract.

The alpha-flip symmetry embedding (evaluating the network on both the airfoil
and its flipped image, then fusing) makes certain identities hold to machine
precision by construction, not just approximately. These tests would catch any
regression in the flip/unflip index bookkeeping in main.py.
"""

import numpy as np
import pytest

import neuralfoil as nf
from neuralfoil._basic_data_type import Data

CAMBERED_KULFAN = dict(
    upper_weights=np.array([0.20, 0.25, 0.20, 0.27, 0.20, 0.25, 0.20, 0.20]),
    lower_weights=np.array([-0.10, -0.05, -0.10, 0.00, 0.05, 0.05, 0.10, 0.10]),
    leading_edge_weight=0.15,
    TE_thickness=0.002,
)


def mirror_kulfan(kulfan: dict) -> dict:
    """Returns the Kulfan parameters of the airfoil mirrored about the chordline."""
    return dict(
        upper_weights=-kulfan["lower_weights"],
        lower_weights=-kulfan["upper_weights"],
        leading_edge_weight=-kulfan["leading_edge_weight"],
        TE_thickness=kulfan["TE_thickness"],
    )


def test_symmetric_airfoil_at_zero_alpha_is_exactly_zero():
    """A symmetric airfoil at alpha=0 must give exactly CL = CM = 0."""
    w = np.array([0.15, 0.20, 0.15, 0.20, 0.15, 0.20, 0.15, 0.15])
    symmetric_kulfan = dict(
        upper_weights=w,
        lower_weights=-w,
        leading_edge_weight=0.0,
        TE_thickness=0.002,
    )
    for model_size in ["medium", "xlarge"]:
        aero = nf.get_aero_from_kulfan_parameters(
            symmetric_kulfan, alpha=0.0, Re=1e6, model_size=model_size
        )
        assert abs(float(aero["CL"][0])) < 1e-14
        assert abs(float(aero["CM"][0])) < 1e-14
        assert float(aero["Top_Xtr"][0]) == pytest.approx(
            float(aero["Bot_Xtr"][0]), abs=1e-14
        )


def test_mirror_identity():
    """
    Evaluating the mirrored airfoil at -alpha must be exactly equal-and-opposite:
    odd outputs (CL, CM) flip sign, even outputs (CD, confidence) are unchanged,
    and upper/lower-surface quantities swap.
    """
    alpha = np.linspace(-8, 12, 5)
    kwargs = dict(Re=5e5, n_crit=8.0, model_size="large")

    aero = nf.get_aero_from_kulfan_parameters(
        CAMBERED_KULFAN, alpha=alpha, xtr_upper=0.7, xtr_lower=0.4, **kwargs
    )
    aero_mirror = nf.get_aero_from_kulfan_parameters(
        mirror_kulfan(CAMBERED_KULFAN), alpha=-alpha, xtr_upper=0.4, xtr_lower=0.7, **kwargs
    )

    tol = dict(rtol=0, atol=1e-12)
    np.testing.assert_allclose(aero_mirror["CL"], -aero["CL"], **tol)
    np.testing.assert_allclose(aero_mirror["CM"], -aero["CM"], **tol)
    np.testing.assert_allclose(aero_mirror["CD"], aero["CD"], **tol)
    np.testing.assert_allclose(
        aero_mirror["analysis_confidence"], aero["analysis_confidence"], **tol
    )
    np.testing.assert_allclose(aero_mirror["Top_Xtr"], aero["Bot_Xtr"], **tol)
    np.testing.assert_allclose(aero_mirror["Bot_Xtr"], aero["Top_Xtr"], **tol)

    for i in range(Data.N):
        np.testing.assert_allclose(
            aero_mirror[f"upper_bl_H_{i}"], aero[f"lower_bl_H_{i}"], **tol
        )
        np.testing.assert_allclose(
            aero_mirror[f"upper_bl_theta_{i}"], aero[f"lower_bl_theta_{i}"], **tol
        )
        np.testing.assert_allclose(
            aero_mirror[f"upper_bl_ue/vinf_{i}"], -aero[f"lower_bl_ue/vinf_{i}"], **tol
        )


def test_output_contract():
    """All advertised keys present, with values in their guaranteed ranges."""
    aero = nf.get_aero_from_kulfan_parameters(
        CAMBERED_KULFAN, alpha=3.0, Re=1e6, model_size="large"
    )

    scalar_keys = ["analysis_confidence", "CL", "CD", "CM", "Top_Xtr", "Bot_Xtr"]
    bl_keys = [
        f"{side}_bl_{quantity}_{i}"
        for side in ["upper", "lower"]
        for quantity in ["theta", "H", "ue/vinf"]
        for i in range(Data.N)
    ]
    assert set(aero.keys()) == set(scalar_keys + bl_keys)

    for key in aero:
        assert np.all(np.isfinite(aero[key])), key
    assert 0 <= float(aero["analysis_confidence"][0]) <= 1
    assert float(aero["CD"][0]) > 0
    assert 0 <= float(aero["Top_Xtr"][0]) <= 1
    assert 0 <= float(aero["Bot_Xtr"][0]) <= 1
    for i in range(Data.N):
        assert float(aero[f"upper_bl_theta_{i}"][0]) > 0
        assert float(aero[f"lower_bl_theta_{i}"][0]) > 0
        assert float(aero[f"upper_bl_H_{i}"][0]) > 1
        assert float(aero[f"lower_bl_H_{i}"][0]) > 1


def test_vectorized_matches_scalar_loop():
    """A vectorized batch must give the same answers as running cases one at a time."""
    alphas = np.array([-10.0, -2.0, 3.0, 8.0, 15.0])
    Res = np.array([5e4, 2e5, 1e6, 5e6, 2e7])

    aero_batch = nf.get_aero_from_kulfan_parameters(
        CAMBERED_KULFAN, alpha=alphas, Re=Res, model_size="medium"
    )
    for i, (alpha, Re) in enumerate(zip(alphas, Res)):
        aero_single = nf.get_aero_from_kulfan_parameters(
            CAMBERED_KULFAN, alpha=alpha, Re=Re, model_size="medium"
        )
        for key in ["analysis_confidence", "CL", "CD", "CM", "Top_Xtr", "Bot_Xtr"]:
            np.testing.assert_allclose(
                aero_batch[key][i], aero_single[key][0], rtol=1e-9, atol=1e-12
            )


def test_bl_x_points_contract():
    """The exported BL sensor locations: N points, strictly increasing, within (0, 1)."""
    x = nf.bl_x_points
    assert len(x) == Data.N
    assert np.all(np.diff(x) > 0)
    assert np.all((x > 0) & (x < 1))


if __name__ == "__main__":
    pytest.main([__file__])
