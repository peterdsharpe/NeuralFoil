"""
The single source of truth for NeuralFoil's latent-space definition.

Everything that must agree between the PyTorch training code and the
NumPy/CasADi inference code is defined here as data:

- the network's input and output columns (names and order),
- the scaling constants that map user-facing quantities to/from latent space,
- how every column behaves under the alpha-flip symmetry.

The signed permutation matrices used to structurally embed the alpha-flip
symmetry are derived programmatically from the column definitions, so the
symmetry bookkeeping is written exactly once. Flipping is a matrix product
(`latent_flipped = latent @ FLIP_MATRIX`), which is backend-universal
(NumPy, CasADi, PyTorch) and autograd-safe.
"""

import numpy as np

from neuralfoil._basic_data_type import Data

N = Data.N  # Number of boundary layer sample points per surface

# Key under which release-artifact .npz files store their JSON provenance
# metadata (written by training/export.py; skipped by the weights loader).
WEIGHTS_METADATA_KEY = "metadata.json"

### Scaling constants between user-facing quantities and latent space.
# These must match the training data preparation exactly; both this module's
# encode/decode helpers and the training pipeline derive from them.
TE_THICKNESS_SCALE = 50.0  # latent = user * scale
LN_RE_CENTER = 12.5  # latent = (ln(Re) - center) / scale
LN_RE_SCALE = 3.5
N_CRIT_CENTER = 9.0  # latent = (n_crit - center) / scale
N_CRIT_SCALE = 4.5
CL_SCALE = 2.0  # latent = user * scale
LN_CD_SCALE = 2.0  # latent = ln(CD) / scale + shift
LN_CD_SHIFT = 2.0
CM_SCALE = 20.0  # latent = user * scale
H_REF = 2.6  # latent = ln(H / ref)
RET_OFFSET = 0.1  # latent = log10(|ue/vinf| * theta * Re + offset)

### Column definitions.
# Each column is a (name, flip_source, flip_sign) triple: under the alpha-flip
# symmetry, the column takes the value `flip_sign * latent[flip_source]`.
# Order matters: it defines the network's input/output vector layout and the
# column order of the scaled training data.

INPUT_COLUMNS = (
    # The flipped airfoil's upper surface is the original's lower surface,
    # mirrored about the chordline (hence the sign flip on CST weights).
    [(f"kulfan_upper_{i}", f"kulfan_lower_{i}", -1.0) for i in range(8)]
    + [(f"kulfan_lower_{i}", f"kulfan_upper_{i}", -1.0) for i in range(8)]
    + [
        ("kulfan_LE_weight", "kulfan_LE_weight", -1.0),
        ("kulfan_TE_thickness", "kulfan_TE_thickness", 1.0),
        ("sin_2a", "sin_2a", -1.0),  # sin(-2a) = -sin(2a)
        ("cos_a", "cos_a", 1.0),  # cos(-a) = cos(a)
        ("1mcos2_a", "1mcos2_a", 1.0),  # sin^2(-a) = sin^2(a)
        ("ln_Re", "ln_Re", 1.0),
        ("n_crit", "n_crit", 1.0),
        ("xtr_upper", "xtr_lower", 1.0),
        ("xtr_lower", "xtr_upper", 1.0),
    ]
)

OUTPUT_COLUMNS = (
    [
        ("analysis_confidence", "analysis_confidence", 1.0),  # even
        ("CL", "CL", -1.0),  # odd
        ("ln_CD", "ln_CD", 1.0),  # even
        ("CM", "CM", -1.0),  # odd
        ("Top_Xtr", "Bot_Xtr", 1.0),  # upper/lower swap
        ("Bot_Xtr", "Top_Xtr", 1.0),
    ]
    + [(f"upper_bl_ret_{i}", f"lower_bl_ret_{i}", 1.0) for i in range(N)]
    + [(f"upper_bl_H_{i}", f"lower_bl_H_{i}", 1.0) for i in range(N)]
    # ue/vinf is signed (negative in reversed flow), so it is odd under flip:
    + [(f"upper_bl_ue/vinf_{i}", f"lower_bl_ue/vinf_{i}", -1.0) for i in range(N)]
    + [(f"lower_bl_ret_{i}", f"upper_bl_ret_{i}", 1.0) for i in range(N)]
    + [(f"lower_bl_H_{i}", f"upper_bl_H_{i}", 1.0) for i in range(N)]
    + [(f"lower_bl_ue/vinf_{i}", f"upper_bl_ue/vinf_{i}", -1.0) for i in range(N)]
)

N_INPUTS = len(INPUT_COLUMNS)  # 25
N_OUTPUTS = len(OUTPUT_COLUMNS)  # 198

INPUT_NAMES = [name for name, _, _ in INPUT_COLUMNS]
OUTPUT_NAMES = [name for name, _, _ in OUTPUT_COLUMNS]


def _flip_matrix(columns: list[tuple[str, str, float]]) -> np.ndarray:
    """
    Builds the signed permutation matrix P for the alpha-flip symmetry,
    such that `latent_flipped = latent @ P` (with latent of shape
    (N_cases, n_columns)).
    """
    name_to_index = {name: i for i, (name, _, _) in enumerate(columns)}
    P = np.zeros((len(columns), len(columns)))
    for j, (_, flip_source, flip_sign) in enumerate(columns):
        P[name_to_index[flip_source], j] = flip_sign
    return P


INPUT_FLIP_MATRIX = _flip_matrix(INPUT_COLUMNS)
OUTPUT_FLIP_MATRIX = _flip_matrix(OUTPUT_COLUMNS)
