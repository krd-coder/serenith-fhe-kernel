"""Cleartext references and opt-in full CPU encrypted decision tests."""
import os
import numpy as np
import pytest

from serenith_demo import quantization as q, reference_math as ref


@pytest.mark.parametrize("dimension", [64])
def test_quantized_reference_and_error(dimension):
    rng = np.random.default_rng(20261022 + dimension)
    a, b = q.normalize(rng.normal(size=dimension)), q.normalize(rng.normal(size=dimension))
    qa, qb = q.quantize(a), q.quantize(b)
    expected = sum(int(x) * int(y) for x, y in zip(qa, qb))
    assert ref.integer_dot(qa, qb) == expected
    assert ref.integer_squared_norm(qa) == sum(int(x) ** 2 for x in qa)
    # For unit vectors and nearest rounding: error <= sqrt(d)/S + d/(4*S²).
    bound = np.sqrt(dimension) / q.SCALE + dimension / (4 * q.SCALE ** 2) + 1e-6
    assert abs(ref.float_similarity(a, b) - ref.quantized_similarity(qa, qb)) <= bound


@pytest.mark.skipif(os.getenv("SERENITH_RUN_FHE") != "1", reason="slow CPU FHE integration")
@pytest.mark.parametrize("dimension", [64])
def test_encrypted_decisions(dimension):
    native = pytest.importorskip("serenith_demo.serenith_tfhe")
    context = native.CpuContext(dimension)
    a = np.zeros(dimension, dtype=np.int8); a[0] = 127
    b = a.copy()
    ca, cb = context.encrypt(a.tolist()), context.encrypt(b.tolist())
    threshold = q.scale_threshold(4, 5)
    assert context.decrypt(context.evaluate(ca, cb, threshold)) == ref.meets_threshold(a, b, 4, 5)
    b[0] = -127
    cb = context.encrypt(b.tolist())
    assert context.decrypt(context.evaluate(ca, cb, threshold)) == ref.meets_threshold(a, b, 4, 5)
    with pytest.raises(ValueError):
        context.encrypt([0] * (dimension - 1))
    bad = [0] * dimension; bad[0] = -128
    with pytest.raises(ValueError):
        context.encrypt(bad)
    with pytest.raises(ValueError):
        context.evaluate(ca, cb, -1)
    with pytest.raises(TypeError):
        context.decrypt(ca)

# TODO: Expose and verify encrypted squared norms if included in the demo API.
# The current API returns only the threshold decision, not encrypted norms or scores.
