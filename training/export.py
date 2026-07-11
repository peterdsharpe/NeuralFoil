"""
Exports a training checkpoint to a NeuralFoil release artifact.

A checkpoint (.pth, torch.save format) is resumable training state: model
weights plus optimizer moments. A release artifact (.npz) is what ships in
the package: just the model weights, framework-neutral and loadable with
NumPy alone, plus embedded JSON provenance metadata (source checkpoint, git
SHA, architecture, timestamp).

Checkpoints are local/ephemeral and gitignored; only exported artifacts are
committed (in neuralfoil/nn_weights_and_biases/). train.py exports an
artifact next to its checkpoint after every epoch, so a current artifact
always exists without manual conversion; promoting one into the package is
the explicit release step:

    uv run python training/export.py nn-xxxlarge.pth --install

After installing new weights, regenerate the golden test fixtures (the
change in shipped behavior is intentional, so re-pin it):

    uv run python tests/fixtures/generate.py
"""

import argparse
import datetime
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from neuralfoil import _spec

PACKAGE_WEIGHTS_DIR = Path(__file__).parent.parent / "neuralfoil" / "nn_weights_and_biases"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def export_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    extra_metadata: dict | None = None,
) -> dict:
    """
    Converts a training checkpoint into a release artifact at `output_path`.

    Returns the embedded metadata dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    arrays = {key: value.cpu().numpy() for key, value in state_dict.items()}
    for key in arrays:
        if not (key.startswith("net.") and key.endswith((".weight", ".bias"))):
            raise ValueError(
                f"Unexpected key {key!r} in checkpoint model_state_dict; "
                f"expected only 'net.{{i}}.weight'/'net.{{i}}.bias' keys."
            )

    metadata = {
        "format_version": 1,
        "generated_by": "training/export.py",
        "source_checkpoint": checkpoint_path.name,
        "git_sha": _git_sha(),
        "export_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        ),
        "n_inputs": _spec.N_INPUTS,
        "n_outputs": _spec.N_OUTPUTS,
        "layer_shapes": {
            key: list(value.shape) for key, value in arrays.items() if "weight" in key
        },
        **(extra_metadata or {}),
    }

    np.savez_compressed(
        output_path,
        **arrays,
        **{_spec.WEIGHTS_METADATA_KEY: np.array(json.dumps(metadata))},
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        nargs="*",
        type=Path,
        default=None,
        help="Checkpoint files to export (default: all nn-*.pth in this directory).",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help=f"Write artifacts into the package ({PACKAGE_WEIGHTS_DIR}) instead of next to each checkpoint.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Free-text provenance note to embed in the artifact metadata.",
    )
    args = parser.parse_args()

    checkpoints = args.checkpoints or sorted(Path(__file__).parent.glob("nn-*.pth"))
    if not checkpoints:
        parser.error("No checkpoints given and no nn-*.pth files found here.")

    for checkpoint_path in checkpoints:
        output_dir = PACKAGE_WEIGHTS_DIR if args.install else checkpoint_path.parent
        output_path = output_dir / checkpoint_path.with_suffix(".npz").name
        metadata = export_checkpoint(
            checkpoint_path,
            output_path,
            extra_metadata={"note": args.note} if args.note else None,
        )
        print(f"{checkpoint_path} -> {output_path}")
        print(f"  {json.dumps(metadata, indent=2)[:300]}...")
