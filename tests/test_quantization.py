"""Boundary and invalid-input tests for quantization."""
import numpy as np
import pytest

from serenith_demo import quantization as q


@pytest.mark.parametrize("dimension", [64])
def test_normalize_and_quantize_boundaries(dimension):
    vector = np.zeros(dimension)
    vector[0] = -3
    result = q.quantize(q.normalize(vector))
    assert result.dtype == np.int8
    assert result[0] == -127
    assert np.count_nonzero(result) == 1


def test_rounding_and_threshold():
    v = np.zeros(64)
    v[0], v[1] = 0.5 / 127, 1.5 / 127
    v[2] = np.sqrt(1 - v[0] ** 2 - v[1] ** 2)
    assert q.quantize(v)[:2].tolist() == [0, 2]
    assert q.scale_threshold(4, 5) == 12904
    assert q.scale_threshold(0, 1) == 0
    assert q.scale_threshold(1, 1) == 16129


@pytest.mark.parametrize("vector", [np.zeros(64), np.full(64, np.nan),
    np.full(64, np.inf), np.ones(128), np.ones((64, 1)),
    np.ones(64, dtype=complex), ["1"] * 64, [True] * 64])
def test_invalid_vectors(vector):
    with pytest.raises(ValueError):
        q.normalize(vector)


@pytest.mark.parametrize("n,d", [(1, 0), (2, 1), (True, 1), (1, 1_000_001), (-1, 2)])
def test_invalid_threshold(n, d):
    with pytest.raises(ValueError):
        q.scale_threshold(n, d)
