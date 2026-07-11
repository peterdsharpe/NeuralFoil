"""
Backend-agnostic implementation of the NeuralFoil model core.

This module is the single implementation of the model mathematics - input
encoding, the MLP forward pass, the alpha-flip symmetry fusion (including the
Mahalanobis analysis-confidence correction), and output decoding. All
quantities and layouts are defined by `neuralfoil._spec`.

Every function takes an array-namespace argument `backend`, so the exact same
code serves:

- inference on NumPy arrays and CasADi symbolics
  (``backend=neuralfoil._backend_numpy``, the default, which dispatches via
  aerosandbox.numpy), and
- training on PyTorch tensors (``backend=neuralfoil._backend_torch``),
  where autograd flows through untouched.

The backend must provide numpy-style: transpose, reshape, concatenate, clip,
exp, log, abs, sum, sind, cosd, swish, and a ``constant(value, like)`` hook
that converts a NumPy constant to the backend's array type (identity for
NumPy/CasADi; dtype/device-matching tensor for torch). Arithmetic operators
(including ``@`` and ``**``) are assumed to work natively on backend arrays.
"""

import aerosandbox.numpy as _asb_numpy

from neuralfoil import _backend_numpy as _numpy_backend
from neuralfoil import _spec

# Small epsilon used to clip sigmoid inputs to suppress overflow; a dynamic
# way to avoid explicitly referring to float bit-widths. (Note: computed for
# float64; the sigmoid is only used at inference-time decoding, not training.)
_eps: float = 10 / _asb_numpy.finfo(_asb_numpy.array(1.0).dtype).max
_ln_eps: float = float(_asb_numpy.log(_eps))


def sigmoid(x, backend=_numpy_backend):
    x = backend.clip(x, _ln_eps, -_ln_eps)  # Clip to suppress overflow
    return 1 / (1 + backend.exp(-x))


def encode_inputs(
    kulfan_parameters,
    alpha,
    Re,
    n_crit,
    xtr_upper,
    xtr_lower,
    backend=_numpy_backend,
) -> list:
    """
    Transforms user-facing inputs into the rows of the input latent space, in
    `_spec.INPUT_COLUMNS` order. Rows are returned unstacked (each row is a
    scalar or a vector of case values), since vectorization/broadcasting
    policy is the caller's concern.
    """
    return [
        *[kulfan_parameters["upper_weights"][i] for i in range(8)],
        *[kulfan_parameters["lower_weights"][i] for i in range(8)],
        kulfan_parameters["leading_edge_weight"],
        kulfan_parameters["TE_thickness"] * _spec.TE_THICKNESS_SCALE,
        backend.sind(2 * alpha),
        backend.cosd(alpha),
        1 - backend.cosd(alpha) ** 2,
        (backend.log(Re) - _spec.LN_RE_CENTER) / _spec.LN_RE_SCALE,
        # No mach input in this architecture generation
        (n_crit - _spec.N_CRIT_CENTER) / _spec.N_CRIT_SCALE,
        xtr_upper,
        xtr_lower,
    ]


def net_forward(x, layer_weights_and_biases: list, backend=_numpy_backend):
    """
    Evaluates the raw MLP (taking in scaled inputs and returning scaled
    outputs). Works entirely in the input and output latent spaces.

    Args:
        x: Input latent array, shape (N_cases, N_inputs).
        layer_weights_and_biases: An ordered list of (weight, bias) pairs,
            one per linear layer.

    Returns: Output latent array, shape (N_cases, N_outputs).
    """
    x = backend.transpose(x)
    n_layers = len(layer_weights_and_biases)
    for i, (w, b) in enumerate(layer_weights_and_biases):
        x = w @ x + backend.reshape(b, (-1, 1))
        if i != n_layers - 1:  # Don't apply the activation on the last layer
            x = backend.swish(x)
    return backend.transpose(x)


def squared_mahalanobis_distance(x, input_distribution: dict, backend=_numpy_backend):
    """
    Computes the squared Mahalanobis distance of query points from the
    training data distribution, in the input latent space.

    Args:
        x: Query points, shape (N_cases, N_inputs).
        input_distribution: dict with "mean_inputs_scaled" (N_inputs,) and
            "inv_cov_inputs_scaled" (N_inputs, N_inputs), as arrays of the
            backend's type.

    Returns: Squared Mahalanobis distances, shape (N_cases,).
    """
    mean = backend.reshape(input_distribution["mean_inputs_scaled"], (1, -1))
    x_minus_mean = backend.transpose(backend.transpose(x) - backend.transpose(mean))
    return backend.sum(
        x_minus_mean @ input_distribution["inv_cov_inputs_scaled"] * x_minus_mean,
        axis=1,
    )


def _with_confidence_correction(y, x, input_distribution, backend):
    """
    Subtracts the squared-Mahalanobis-distance term from the analysis
    confidence logit (column 0 of the output latent space). This is baked into
    training as well as inference, to ensure the network asymptotes to zero
    analysis confidence far away from the training data.
    """
    correction = squared_mahalanobis_distance(x, input_distribution, backend) / (
        2 * _spec.N_INPUTS
    )
    return backend.concatenate(
        [y[:, 0:1] - backend.reshape(correction, (-1, 1)), y[:, 1:]],
        axis=1,
    )


def evaluate_fused(
    x,
    layer_weights_and_biases: list,
    input_distribution: dict,
    backend=_numpy_backend,
):
    """
    The full latent-space model evaluation, shared verbatim between training
    and inference: evaluates the network on both the query point and its
    alpha-flipped image, un-flips the latter, and averages the two. This
    structurally embeds the alpha-flip symmetry of airfoil physics.

    Returns the fused output latent array, shape (N_cases, N_outputs), with
    transition locations clipped to [0, 1] and the Mahalanobis correction
    applied to the analysis-confidence logit (column 0). The logit is NOT
    passed through a sigmoid here, so that training can use a
    binary-cross-entropy-with-logits loss; inference applies the sigmoid
    during decoding.
    """
    P_in = backend.constant(_spec.INPUT_FLIP_MATRIX, like=x)
    P_out = backend.constant(_spec.OUTPUT_FLIP_MATRIX, like=x)

    ### Evaluate the network normally
    y = net_forward(x, layer_weights_and_biases, backend)
    y = _with_confidence_correction(y, x, input_distribution, backend)

    ### Then, evaluate it on the alpha-flipped inputs, and un-flip the result
    x_flipped = x @ P_in
    y_flipped = net_forward(x_flipped, layer_weights_and_biases, backend)
    y_flipped = _with_confidence_correction(
        y_flipped, x_flipped, input_distribution, backend
    )
    y_unflipped = y_flipped @ P_out

    ### Average the two evaluations to get the symmetric result
    y_fused = (y + y_unflipped) / 2

    ### Clip transition locations to their physical range
    return backend.concatenate(
        [
            y_fused[:, 0:4],
            backend.clip(y_fused[:, 4:5], 0, 1),  # Top_Xtr
            backend.clip(y_fused[:, 5:6], 0, 1),  # Bot_Xtr
            y_fused[:, 6:],
        ],
        axis=1,
    )


def decode_outputs(y_fused, Re, backend=_numpy_backend) -> dict:
    """
    Transforms the fused output latent array into a dictionary of user-facing
    outputs (`Data.get_vector_output_column_names()` keys). `Re` is needed to
    reconstruct the boundary layer momentum thickness from the latent
    momentum-thickness Reynolds number.
    """
    N = _spec.N

    upper_bl_ue_over_vinf = y_fused[:, 6 + N * 2 : 6 + N * 3]
    lower_bl_ue_over_vinf = y_fused[:, 6 + N * 5 : 6 + N * 6]

    Re_column = backend.reshape(Re, (-1, 1))
    upper_theta = ((10 ** y_fused[:, 6 : 6 + N]) - _spec.RET_OFFSET) / (
        backend.abs(upper_bl_ue_over_vinf) * Re_column
    )
    upper_H = _spec.H_REF * backend.exp(y_fused[:, 6 + N : 6 + N * 2])

    lower_theta = ((10 ** y_fused[:, 6 + N * 3 : 6 + N * 4]) - _spec.RET_OFFSET) / (
        backend.abs(lower_bl_ue_over_vinf) * Re_column
    )
    lower_H = _spec.H_REF * backend.exp(y_fused[:, 6 + N * 4 : 6 + N * 5])

    return {
        "analysis_confidence": sigmoid(y_fused[:, 0], backend),
        "CL": y_fused[:, 1] / _spec.CL_SCALE,
        "CD": backend.exp((y_fused[:, 2] - _spec.LN_CD_SHIFT) * _spec.LN_CD_SCALE),
        "CM": y_fused[:, 3] / _spec.CM_SCALE,
        "Top_Xtr": y_fused[:, 4],
        "Bot_Xtr": y_fused[:, 5],
        **{f"upper_bl_theta_{i}": upper_theta[:, i] for i in range(N)},
        **{f"upper_bl_H_{i}": upper_H[:, i] for i in range(N)},
        **{f"upper_bl_ue/vinf_{i}": upper_bl_ue_over_vinf[:, i] for i in range(N)},
        **{f"lower_bl_theta_{i}": lower_theta[:, i] for i in range(N)},
        **{f"lower_bl_H_{i}": lower_H[:, i] for i in range(N)},
        **{f"lower_bl_ue/vinf_{i}": lower_bl_ue_over_vinf[:, i] for i in range(N)},
    }
