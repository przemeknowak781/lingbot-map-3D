"""E1 validation harness — measure absolute metric accuracy of a 3D reconstruction.

The one question this package exists to answer: on a real, calibrated small
object, how far (in millimetres) is a surface-aligned Gaussian-Splatting
reconstruction from ground truth — and does it clear the target vertical's
tolerance? See the decision document, edge E1.
"""

from .config import Config
from .pipeline import run

__all__ = ["Config", "run"]
__version__ = "0.1.0"
