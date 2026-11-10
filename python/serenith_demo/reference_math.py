"""Cleartext references for similarity and quantized threshold decisions."""

import numpy as np
from numpy.typing import ArrayLike

from .quantization import SCALE, real_vector, scale_threshold, validate_dimension


def float_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Return cosine similarity for nonzero vectors, using float32 arithmetic."""
    x, y = real_vector(a).astype(np.float32), real_vector(b).astype(np.float32)
    if x.shape != y.shape:
        raise ValueError("vector dimensions must match")
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if not np.isfinite(nx) or not np.isfinite(ny) or nx == 0 or ny == 0:
        raise ValueError("vectors must have finite nonzero norms")
    return float(np.dot(x / nx, y / ny))


def integer_dot(a: ArrayLike, b: ArrayLike) -> int:
    """Dot product of quantized int8 vectors with int64 accumulation."""
    x, y = np.asarray(a), np.asarray(b)
    if x.ndim != 1 or y.ndim != 1 or x.dtype != np.int8 or y.dtype != np.int8:
        raise ValueError("vectors must be one-dimensional int8 arrays")
    validate_dimension(x.size)
    if x.shape != y.shape or np.any(x == -128) or np.any(y == -128):
        raise ValueError("vectors must match and contain values in [-127, 127]")
    return int(np.dot(x.astype(np.int64), y.astype(np.int64)))


def integer_squared_norm(a: ArrayLike) -> int:
    """Return the squared L2 norm in the quantized integer domain."""
    return integer_dot(a, a)


def quantized_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Approximate normalized similarity by rescaling the integer dot product."""
    return integer_dot(a, b) / (SCALE * SCALE)


def meets_threshold(a: ArrayLike, b: ArrayLike, numerator: int, denominator: int) -> bool:
    """Compare the quantized dot product to the PoC-scaled threshold."""
    return integer_dot(a, b) >= scale_threshold(numerator, denominator)
