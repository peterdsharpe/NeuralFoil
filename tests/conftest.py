import sys
from pathlib import Path

# Make the repo root importable, so that tests can import the (unpackaged)
# training code, e.g. `from training.model import Net`. Tests that need this
# guard with pytest.importorskip, since the training code (and torch) are not
# present in all environments where the test suite runs.
sys.path.insert(0, str(Path(__file__).parent.parent))
