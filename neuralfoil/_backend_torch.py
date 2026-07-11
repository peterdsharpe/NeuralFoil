"""
PyTorch binding of the minimal array-namespace protocol used by
`neuralfoil._core`.

This module is only imported by the training code (and cross-backend tests);
nothing in the inference path imports it, so NeuralFoil keeps its
zero-torch-dependency guarantee. Autograd flows through all of these ops.
"""

import torch


def transpose(x):
    return x.T  # The core only transposes 2D arrays


def reshape(x, shape):
    return torch.reshape(x, shape)


def concatenate(arrays, axis=0):
    return torch.cat(arrays, dim=axis)


def clip(x, lower, upper):
    return torch.clamp(x, lower, upper)


def exp(x):
    return torch.exp(x)


def log(x):
    return torch.log(x)


def abs(x):
    return torch.abs(x)


def sum(x, axis=None):
    return torch.sum(x, dim=axis)


def sind(x):
    return torch.sin(torch.deg2rad(x))


def cosd(x):
    return torch.cos(torch.deg2rad(x))


def swish(x):
    return torch.nn.functional.silu(x)


# The flip matrices are fixed module-level constants, so cache their tensor
# conversions per (array, dtype, device) rather than re-copying (potentially
# host-to-device) on every forward pass.
_constant_cache: dict = {}


def constant(value, like):
    """Converts a NumPy constant to a tensor matching `like`'s dtype and device."""
    key = (id(value), like.dtype, like.device)
    if key not in _constant_cache:
        _constant_cache[key] = torch.as_tensor(
            value, dtype=like.dtype, device=like.device
        )
    return _constant_cache[key]
