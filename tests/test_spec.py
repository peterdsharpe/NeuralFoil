"""
Tests of the latent-space spec (`neuralfoil._spec`).

The critical test here is `test_flip_matrices_match_legacy_logic`: it checks
the programmatically-derived flip matrices against a verbatim copy of the
hand-written index bookkeeping that previously lived (twice) in main.py and
train.py. That legacy code is preserved below as the executable definition of
the gen-2 behavior the matrices must reproduce.
"""

import numpy as np
import pytest

from neuralfoil import _spec


def test_column_counts_and_uniqueness():
    assert _spec.N_INPUTS == 25
    assert _spec.N_OUTPUTS == 6 + 6 * _spec.N
    assert len(set(_spec.INPUT_NAMES)) == _spec.N_INPUTS
    assert len(set(_spec.OUTPUT_NAMES)) == _spec.N_OUTPUTS
    for column in _spec.INPUT_COLUMNS + _spec.OUTPUT_COLUMNS:
        assert column.flip_source in (_spec.INPUT_NAMES + _spec.OUTPUT_NAMES), column.name
        assert column.flip_sign in (-1.0, 1.0), column.name


@pytest.mark.parametrize(
    "P", [_spec.INPUT_FLIP_MATRIX, _spec.OUTPUT_FLIP_MATRIX], ids=["input", "output"]
)
def test_flip_is_an_involution(P):
    """Flipping twice must be the identity."""
    np.testing.assert_array_equal(P @ P, np.eye(len(P)))
    # ...and each column must have exactly one nonzero entry, of magnitude 1:
    assert np.all(np.sum(np.abs(P), axis=0) == 1)
    assert np.all(np.sum(np.abs(P), axis=1) == 1)


def _legacy_flip_inputs(x: np.ndarray) -> np.ndarray:
    """Verbatim copy of the gen-2 hand-written input flip (from main.py)."""
    x_flipped = x + 0.0
    x_flipped[:, :8] = x[:, 8:16] * -1  # switch kulfan_lower with a flipped kulfan_upper
    x_flipped[:, 8:16] = x[:, :8] * -1  # switch kulfan_upper with a flipped kulfan_lower
    x_flipped[:, 16] = -1 * x[:, 16]  # flip kulfan_LE_weight
    x_flipped[:, 18] = -1 * x[:, 18]  # flip sin(2a)
    x_flipped[:, 23] = x[:, 24]  # flip xtr_upper with xtr_lower
    x_flipped[:, 24] = x[:, 23]  # flip xtr_lower with xtr_upper
    return x_flipped


def _legacy_unflip_outputs(y_flipped: np.ndarray) -> np.ndarray:
    """Verbatim copy of the gen-2 hand-written output unflip (from main.py)."""
    N = _spec.N
    y_unflipped = y_flipped + 0.0
    y_unflipped[:, 1] = y_flipped[:, 1] * -1  # CL
    y_unflipped[:, 3] = y_flipped[:, 3] * -1  # CM
    y_unflipped[:, 4] = y_flipped[:, 5]  # switch Top_Xtr with Bot_Xtr
    y_unflipped[:, 5] = y_flipped[:, 4]  # switch Bot_Xtr with Top_Xtr

    # switch upper and lower Ret, H
    y_unflipped[:, 6 : 6 + N * 2] = y_flipped[:, 6 + N * 3 : 6 + N * 5]
    y_unflipped[:, 6 + N * 3 : 6 + N * 5] = y_flipped[:, 6 : 6 + N * 2]

    # switch upper_bl_ue/vinf with lower_bl_ue/vinf
    y_unflipped[:, 6 + N * 2 : 6 + N * 3] = -1 * y_flipped[:, 6 + N * 5 : 6 + N * 6]
    y_unflipped[:, 6 + N * 5 : 6 + N * 6] = -1 * y_flipped[:, 6 + N * 2 : 6 + N * 3]
    return y_unflipped


def test_flip_matrices_match_legacy_logic():
    rng = np.random.default_rng(0)

    x = rng.standard_normal((17, _spec.N_INPUTS))
    np.testing.assert_array_equal(x @ _spec.INPUT_FLIP_MATRIX, _legacy_flip_inputs(x))

    y = rng.standard_normal((17, _spec.N_OUTPUTS))
    np.testing.assert_array_equal(
        y @ _spec.OUTPUT_FLIP_MATRIX, _legacy_unflip_outputs(y)
    )


def test_output_names_align_with_data_class():
    """
    The spec's output layout must match Data.get_vector_output_column_names()
    positionally (names differ only where latent-space and user-space
    quantities differ: ln_CD vs CD, ret vs theta).
    """
    from neuralfoil._basic_data_type import Data

    data_names = Data.get_vector_output_column_names()
    assert len(data_names) == _spec.N_OUTPUTS
    for spec_name, data_name in zip(_spec.OUTPUT_NAMES, data_names):
        assert (
            spec_name == data_name
            or (spec_name, data_name) == ("ln_CD", "CD")
            or spec_name.replace("_ret_", "_theta_") == data_name
        ), (spec_name, data_name)


if __name__ == "__main__":
    pytest.main([__file__])
