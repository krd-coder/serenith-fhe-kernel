"""Adapter between the linear demo and the in-memory CPU Rust extension."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


class NativeBackend:
    """Single-key CPU or CUDA backend; no external ciphertexts."""

    def __init__(self, show_progress: bool = False, backend: str = "cpu") -> None:
        if backend not in ("cpu", "cuda"):
            raise ValueError("backend must be cpu or cuda")
        if backend == "cuda":
            from .serenith_tfhe import cuda_compiled
            if not cuda_compiled():
                raise RuntimeError("CUDA support is not enabled; run bash scripts/setup.sh --cuda to rebuild")
        self.show_progress = show_progress
        self.backend = backend

    def setup(self, dimension: int) -> Any:
        from .serenith_tfhe import CpuContext
        context = CpuContext(dimension, self.backend)
        context.set_progress(self.show_progress)
        return context

    def encrypt(self, context: Any, vector: NDArray[np.int8], prove: bool) -> Any:
        return context.encrypt(vector.tolist(), prove)

    def verify(self, context: Any, ciphertext: Any) -> bool:
        return context.verify(ciphertext)

    def evaluate(self, context: Any, a: Any, b: Any, scaled_threshold: int) -> Any:
        return context.evaluate(a, b, scaled_threshold)

    def decrypt(self, context: Any, result: Any) -> bool:
        return context.decrypt(result)
