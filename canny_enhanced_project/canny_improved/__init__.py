"""Improved, experimentally evaluated version of the manual Canny edge detector."""
from .pipeline import CannyResult, canny_improved
from .stages import to_gray

__all__ = ["CannyResult", "canny_improved", "to_gray"]
