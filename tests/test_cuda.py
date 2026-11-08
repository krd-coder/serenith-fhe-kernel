"""Opt-in 64D CUDA integration: requires a CUDA-enabled build and NVIDIA GPU."""
import os
import pytest

from serenith_demo.quantization import scale_threshold


def test_cpu_only_build_rejects_cuda_explicitly():
    native = pytest.importorskip("serenith_demo.serenith_tfhe")
    if native.cuda_compiled():
        pytest.skip("test applies to a CPU-only extension")
    with pytest.raises(ValueError, match="CUDA support is not enabled"):
        native.CpuContext(64, "cuda")
    from serenith_demo.native_backend import NativeBackend
    with pytest.raises(RuntimeError, match="CUDA support is not enabled"):
        NativeBackend(backend="cuda")


@pytest.mark.skipif(os.environ.get("SERENITH_RUN_CUDA") != "1", reason="requires CUDA GPU")
def test_cuda_decisions_64():
    native = pytest.importorskip("serenith_demo.serenith_tfhe")
    assert native.cuda_compiled(), "extension must be compiled with the cuda feature"
    ctx = native.CpuContext(64, "cuda")
    left = [0] * 64
    right = [0] * 64
    left[0] = right[0] = 127
    a, b = ctx.encrypt(left), ctx.encrypt(right)
    assert ctx.decrypt(ctx.evaluate(a, b, scale_threshold(4, 5))) is True
    right[0] = -127
    negative = ctx.encrypt(right)
    assert ctx.decrypt(ctx.evaluate(a, negative, scale_threshold(4, 5))) is False


@pytest.mark.skipif(os.environ.get("SERENITH_RUN_CUDA_PROOF") != "1", reason="requires CUDA GPU and ZKV2")
def test_cuda_proven_ciphertexts_64():
    native = pytest.importorskip("serenith_demo.serenith_tfhe")
    assert native.cuda_compiled(), "extension must be compiled with the cuda feature"
    ctx = native.CpuContext(64, "cuda")
    vector = [0] * 64
    vector[0] = 127
    ciphertext = ctx.encrypt(vector, True)
    assert ctx.verify(ciphertext) is True
    assert ctx.decrypt(ctx.evaluate(ciphertext, ciphertext, scale_threshold(4, 5))) is True
