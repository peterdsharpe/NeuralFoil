"""
Golden-value regression tests.

These pin exact expected outputs for a few fixed cases, with tolerances loose
enough for cross-platform floating-point noise (different BLAS libraries) but
tight enough to catch real behavior changes: weight-file corruption, changes
in the encode/decode transforms, AeroSandbox geometry-pipeline drift, or
numpy behavioral changes.

If `test_golden_kulfan_direct` passes but `test_golden_airfoil_pipeline`
fails, the change is on the AeroSandbox side (geometry normalization or
CST fitting), not in NeuralFoil's own weights or math.
"""

import numpy as np
import pytest

import neuralfoil as nf

# A fixed, mildly-cambered test airfoil, defined directly in Kulfan parameters
# so that no geometry-fitting pipeline is involved.
KULFAN = dict(
    upper_weights=np.array([0.20, 0.25, 0.20, 0.27, 0.20, 0.25, 0.20, 0.20]),
    lower_weights=np.array([-0.10, -0.05, -0.10, 0.00, 0.05, 0.05, 0.10, 0.10]),
    leading_edge_weight=0.15,
    TE_thickness=0.002,
)

GOLDEN_KULFAN = {
    "medium": {
        "analysis_confidence": 0.955711837783,
        "CL": 1.10332809679,
        "CD": 0.00919882438456,
        "CM": -0.110598030451,
        "Top_Xtr": 0.250543496781,
        "Bot_Xtr": 0.964878409058,
    },
    "xxxlarge": {
        "analysis_confidence": 0.971352552774,
        "CL": 1.09450207214,
        "CD": 0.00911818948927,
        "CM": -0.108654837628,
        "Top_Xtr": 0.253019285687,
        "Bot_Xtr": 1.0,
    },
}

GOLDEN_NACA4412_XLARGE = {
    "analysis_confidence": 0.968356999039,
    "CL": 1.02508809111,
    "CD": 0.00791137686241,
    "CM": -0.0992982374935,
    "Top_Xtr": 0.402283708087,
    "Bot_Xtr": 1.0,
}


@pytest.mark.parametrize("model_size", GOLDEN_KULFAN.keys())
def test_golden_kulfan_direct(model_size):
    aero = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=KULFAN,
        alpha=5.0,
        Re=1e6,
        model_size=model_size,
    )
    for key, expected in GOLDEN_KULFAN[model_size].items():
        assert float(aero[key][0]) == pytest.approx(expected, rel=1e-6, abs=1e-9), key


def test_golden_airfoil_pipeline():
    asb = pytest.importorskip("aerosandbox")
    aero = nf.get_aero_from_airfoil(
        airfoil=asb.Airfoil("naca4412"),
        alpha=5.0,
        Re=1e6,
        model_size="xlarge",
    )
    for key, expected in GOLDEN_NACA4412_XLARGE.items():
        assert float(aero[key][0]) == pytest.approx(expected, rel=1e-6, abs=1e-9), key


if __name__ == "__main__":
    pytest.main([__file__])
