"""
Cross-backend equivalence tests: the single model implementation in
neuralfoil/_core.py must produce identical results whether evaluated through
the NumPy/CasADi backend (inference) or the PyTorch backend (training).

This is the test that makes "one source of truth" an enforced property rather
than a discipline: any divergence between what is trained and what is shipped
fails here. Skipped automatically in environments without torch (e.g., the
lightweight CI matrix); runs in the dev environment.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralfoil import _backend_torch, _core
from neuralfoil.main import _nn_parameters, _scaled_input_distribution

MODEL_SIZE = "large"  # shapes: (128, 25), 3x (128, 128), (198, 128)
N_HIDDEN_LAYERS = 3
WIDTH = 128


@pytest.fixture(scope="module")
def numpy_weights():
    nn_params = _nn_parameters[MODEL_SIZE]
    layer_indices = sorted({int(key.split(".")[1]) for key in nn_params.keys()})
    return [
        (
            nn_params[f"net.{i}.weight"].astype(np.float64),
            nn_params[f"net.{i}.bias"].astype(np.float64),
        )
        for i in layer_indices
    ]


@pytest.fixture(scope="module")
def numpy_distribution():
    return {
        key: np.asarray(value, dtype=np.float64)
        for key, value in _scaled_input_distribution.items()
    }


@pytest.fixture(scope="module")
def x_batch():
    rng = np.random.default_rng(0)
    return rng.standard_normal((11, 25))


def test_core_torch_matches_numpy_float64(numpy_weights, numpy_distribution, x_batch):
    """Same weights, same inputs, float64 both sides: results must agree to ~machine precision."""
    y_numpy = _core.evaluate_fused(x_batch, numpy_weights, numpy_distribution)

    y_torch = _core.evaluate_fused(
        torch.tensor(x_batch, dtype=torch.float64),
        [
            (torch.tensor(w), torch.tensor(b))
            for w, b in numpy_weights
        ],
        {
            key: torch.tensor(value)
            for key, value in numpy_distribution.items()
        },
        backend=_backend_torch,
    )

    np.testing.assert_allclose(y_torch.numpy(), y_numpy, rtol=1e-12, atol=1e-12)


def test_training_net_matches_numpy_inference(numpy_weights, x_batch):
    """
    The actual training-side Net module (with the shipped weights loaded into
    it) must agree with the NumPy inference path. This exercises the full
    wrapper plumbing: parameter registration/extraction order and the
    Mahalanobis buffers.
    """
    training_model = pytest.importorskip(
        "training.model", reason="training code not present (e.g., running from sdist)"
    )

    net = training_model.Net(
        mean_inputs_scaled=torch.tensor(
            _scaled_input_distribution["mean_inputs_scaled"], dtype=torch.float64
        ),
        cov_inputs_scaled=torch.tensor(
            _scaled_input_distribution["cov_inputs_scaled"], dtype=torch.float64
        ),
        n_hidden_layers=N_HIDDEN_LAYERS,
        width=WIDTH,
    ).double()
    net.load_state_dict(
        {
            key: torch.tensor(value)
            for key, value in _nn_parameters[MODEL_SIZE].items()
        }
    )

    x = torch.tensor(x_batch, dtype=torch.float64, requires_grad=True)
    y_torch = net(x)

    # NumPy side, using the Net's own inverse-covariance so the comparison is
    # exact (the shipped npz stores a separately-precomputed inverse):
    y_numpy = _core.evaluate_fused(
        x_batch,
        numpy_weights,
        {
            "mean_inputs_scaled": net.mean_inputs_scaled.numpy(),
            "inv_cov_inputs_scaled": net.inv_cov_inputs_scaled.numpy(),
        },
    )
    np.testing.assert_allclose(
        y_torch.detach().numpy(), y_numpy, rtol=1e-12, atol=1e-12
    )

    # Autograd must flow through the shared core (matmul-based flips,
    # functional confidence correction):
    y_torch.sum().backward()
    assert x.grad is not None and torch.all(torch.isfinite(x.grad))
    for parameter in net.parameters():
        assert parameter.grad is not None


def test_core_torch_float32_close_to_float64(numpy_weights, numpy_distribution, x_batch):
    """Float32 (the training dtype) agrees with float64 to float32 precision."""
    y64 = _core.evaluate_fused(x_batch, numpy_weights, numpy_distribution)
    y32 = _core.evaluate_fused(
        torch.tensor(x_batch, dtype=torch.float32),
        [
            (torch.tensor(w, dtype=torch.float32), torch.tensor(b, dtype=torch.float32))
            for w, b in numpy_weights
        ],
        {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in numpy_distribution.items()
        },
        backend=_backend_torch,
    )
    np.testing.assert_allclose(y32.numpy(), y64, rtol=2e-3, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
