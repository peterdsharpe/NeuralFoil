"""
Dense golden-fixture regression test.

Compares every output (bulk coefficients + all boundary-layer distributions)
of every model size against the pinned fixture in
`tests/fixtures/gen2_golden.npz`, across a battery of diverse cases including
far-out-of-distribution ones. See `tests/fixtures/generate.py` for the battery
definition and regeneration instructions.

Tolerances are tight enough to catch any real behavior change, while allowing
for cross-platform BLAS float noise.
"""

from pathlib import Path

import numpy as np
import pytest

from fixtures.generate import MODEL_SIZES, evaluate_case

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "gen2_golden.npz"


@pytest.fixture(scope="module")
def fixture():
    return dict(np.load(FIXTURE_PATH))


@pytest.mark.parametrize("model_size", MODEL_SIZES)
def test_against_golden_fixture(fixture, model_size):
    inputs = fixture["inputs"]
    expected = fixture[f"outputs/{model_size}"]
    output_keys = fixture["output_keys"]

    for i, case in enumerate(inputs):
        actual = evaluate_case(list(case), model_size)
        for j, key in enumerate(output_keys):
            np.testing.assert_allclose(
                actual[j],
                expected[i, j],
                rtol=1e-6,
                atol=1e-12,
                err_msg=f"{model_size}, case {i}, output {key}",
            )


if __name__ == "__main__":
    pytest.main([__file__])
