"""
Generates the golden inference fixtures in `gen2_golden.npz`.

The fixture pins the exact input->output behavior of the shipped (gen-2)
models across all 8 sizes and a diverse battery of cases, including every
output (bulk coefficients and all boundary-layer distributions). It is the
safety net for refactors of the inference path: any change to the shipped
behavior, however small, fails `test_golden_fixtures.py`.

Regenerate (only when an intentional behavior change is made) with:

    uv run python tests/fixtures/generate.py
"""

from pathlib import Path

import numpy as np

import neuralfoil as nf
from neuralfoil._basic_data_type import Data

FIXTURE_PATH = Path(__file__).parent / "gen2_golden.npz"

MODEL_SIZES = [
    "xxsmall",
    "xsmall",
    "small",
    "medium",
    "large",
    "xlarge",
    "xxlarge",
    "xxxlarge",
]

# Each case: 8 upper weights, 8 lower weights, LE weight, TE thickness,
# alpha, Re, n_crit, xtr_upper, xtr_lower.
INPUT_COLUMNS = (
    [f"upper_{i}" for i in range(8)]
    + [f"lower_{i}" for i in range(8)]
    + ["leading_edge_weight", "TE_thickness", "alpha", "Re", "n_crit", "xtr_upper", "xtr_lower"]
)

_cambered_upper = [0.20, 0.25, 0.20, 0.27, 0.20, 0.25, 0.20, 0.20]
_cambered_lower = [-0.10, -0.05, -0.10, 0.00, 0.05, 0.05, 0.10, 0.10]
_symmetric = [0.15, 0.20, 0.15, 0.20, 0.15, 0.20, 0.15, 0.15]

CASES = [
    # Typical cambered airfoil, cruise-ish conditions
    _cambered_upper + _cambered_lower + [0.15, 0.002, 5.0, 1e6, 9.0, 1.0, 1.0],
    # Low Re, dirty-tunnel n_crit, negative alpha
    _cambered_upper + _cambered_lower + [0.15, 0.002, -5.0, 5e4, 4.0, 1.0, 1.0],
    # Exactly-symmetric airfoil at alpha=0 (exact-zero lift/moment case)
    _symmetric + [-w for w in _symmetric] + [0.0, 0.002, 0.0, 1e6, 9.0, 1.0, 1.0],
    # Symmetric airfoil, high alpha, forced upper transition
    _symmetric + [-w for w in _symmetric] + [0.0, 0.002, 12.0, 1e7, 9.0, 0.3, 1.0],
    # Thick, blunt, strange airfoil
    [0.45] * 8 + [-0.40] * 8 + [0.40, 0.010, 2.0, 3e5, 9.0, 1.0, 1.0],
    # Very thin airfoil, clean-air n_crit, strongly negative alpha
    [0.05] * 8 + [-0.05] * 8 + [0.0, 0.001, -15.0, 2e6, 14.0, 1.0, 1.0],
    # Reflexed-ish camber line, transitional Re
    _cambered_upper + [-0.10, -0.05, 0.00, 0.05, 0.10, 0.10, -0.05, -0.10] + [0.10, 0.002, 8.0, 8e4, 9.0, 1.0, 1.0],
    # Far out-of-distribution: post-stall
    _cambered_upper + _cambered_lower + [0.15, 0.002, 90.0, 1e6, 9.0, 1.0, 1.0],
    # Far out-of-distribution: reversed flow, Stokes-ish Re
    _cambered_upper + _cambered_lower + [0.15, 0.002, -180.0, 1e3, 9.0, 1.0, 1.0],
    # Very high Re, forced trips near LE on both surfaces
    _cambered_upper + _cambered_lower + [0.15, 0.002, 3.0, 1e9, 9.0, 0.05, 0.05],
]

OUTPUT_KEYS = Data.get_vector_output_column_names()


def evaluate_case(case: list, model_size: str) -> np.ndarray:
    """Runs one battery case through NeuralFoil, returning outputs as a flat vector."""
    aero = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=dict(
            upper_weights=np.array(case[0:8]),
            lower_weights=np.array(case[8:16]),
            leading_edge_weight=case[16],
            TE_thickness=case[17],
        ),
        alpha=case[18],
        Re=case[19],
        n_crit=case[20],
        xtr_upper=case[21],
        xtr_lower=case[22],
        model_size=model_size,
    )
    return np.array([float(aero[key][0]) for key in OUTPUT_KEYS])


if __name__ == "__main__":
    fixture = {
        "inputs": np.array(CASES, dtype=np.float64),
        "output_keys": np.array(OUTPUT_KEYS),
    }
    for model_size in MODEL_SIZES:
        fixture[f"outputs/{model_size}"] = np.stack(
            [evaluate_case(case, model_size) for case in CASES]
        )
        print(f"{model_size}: {fixture[f'outputs/{model_size}'].shape}")

    np.savez_compressed(FIXTURE_PATH, **fixture)
    print(f"Wrote {FIXTURE_PATH}")
