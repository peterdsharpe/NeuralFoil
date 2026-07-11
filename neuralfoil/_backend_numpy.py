"""
NumPy/CasADi binding of the minimal array-namespace protocol used by
`neuralfoil._core`.

aerosandbox.numpy already dispatches every operation to either NumPy or
CasADi based on the input types, so this binding is mostly re-exports. It
exists so that the core's backend protocol is explicit, and so that the one
non-numpy hook (`constant`) has a home.
"""

from aerosandbox.numpy import (  # noqa: F401
    abs,
    clip,
    concatenate,
    cosd,
    exp,
    log,
    reshape,
    sind,
    sum,
    swish,
    transpose,
)


def constant(value, like=None):
    """
    Returns a NumPy constant array as-is: NumPy broadcasting and CasADi's
    operator overloads both consume NumPy arrays directly. (The torch binding
    instead converts to a tensor matching `like`'s dtype and device.)
    """
    return value
