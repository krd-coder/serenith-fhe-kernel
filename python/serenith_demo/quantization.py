"""MRL prefix normalization and signed integer quantization, matching the PoC."""

import math
import numbers

import numpy as np
from numpy.typing import ArrayLike, NDArray

SCALE = 127
SUPPORTED_DIMENSIONS = (64, 256, 768)


def validate_dimension(dimension: int) -> None:
    """Reject dimensions outside the PoC configuration."""
    if type(dimension) is not int or dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError(f"dimension must be one of {SUPPORTED_DIMENSIONS}")


def real_vector(values: ArrayLike) -> NDArray[np.float64]:
    """Validate and convert a supported one-dimensional real vector."""
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError("vector must be a one-dimensional real numeric array")
    validate_dimension(raw.size)
    vector = raw.astype(np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector must contain only finite values")
    return vector


def normalize(values: ArrayLike) -> NDArray[np.float64]:
    """L2-normalize a vector; a zero vector has no cosine direction."""
    vector = real_vector(values)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("vector must have a finite, nonzero L2 norm")
    return vector / norm


def quantize(values: ArrayLike) -> NDArray[np.int8]:
    """Quantize a normalized vector using nearest, ties-to-even rounding."""
    vector = real_vector(values)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or abs(norm - 1.0) >= 1e-5:
        raise ValueError("vector must be L2-normalized")
    result = np.rint(vector * SCALE).astype(np.int64)
    if np.any(np.abs(result) > SCALE):
        raise ValueError("quantized coordinate exceeds the supported bounds")
    return result.astype(np.int8)


def scale_threshold(numerator: int, denominator: int) -> int:
    """Return ceil((numerator / denominator) * SCALE²), as in the PoC."""
    if (not isinstance(numerator, numbers.Integral) or isinstance(numerator, (bool, np.bool_))
            or not isinstance(denominator, numbers.Integral) or isinstance(denominator, (bool, np.bool_))
            or denominator <= 0 or denominator > 1_000_000 or not 0 <= numerator <= denominator):
        raise ValueError("threshold must satisfy 0 <= numerator <= denominator <= 1000000")
    return (int(numerator) * SCALE * SCALE + int(denominator) - 1) // int(denominator)
