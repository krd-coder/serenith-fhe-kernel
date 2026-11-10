"""Linear single-key demo orchestration with an explicit FHE backend contract."""

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from .embedder import DocumentEmbedder
from .quantization import normalize, quantize, scale_threshold, validate_dimension
from .reference_math import float_similarity, meets_threshold, quantized_similarity


class FHEBackend(Protocol):
    """Single-key adapter; verify must authenticate proofs against trusted setup."""

    def setup(self, dimension: int) -> Any: ...
    def encrypt(self, context: Any, vector: NDArray[np.int8], prove: bool) -> Any: ...
    def verify(self, context: Any, ciphertext: Any) -> bool: ...
    def evaluate(self, context: Any, a: Any, b: Any, scaled_threshold: int) -> Any: ...
    def decrypt(self, context: Any, result: Any) -> bool: ...


@dataclass(frozen=True)
class DemoResult:
    """Decrypted evaluation and its independent cleartext references."""

    decision: bool
    reference_decision: bool
    float_similarity: float
    quantized_similarity: float
    quantization_error: float


def run_demo(text_a: str, text_b: str, dimension: int, numerator: int, denominator: int,
             backend: FHEBackend | None = None, *, embedder: DocumentEmbedder | None = None,
             prove: bool = False, progress: Callable[[str], None] | None = None,
             backend_name: str = "cpu") -> DemoResult:
    """Run preprocessing and an injected single-key backend; verify any proofs first."""
    validate_dimension(dimension)
    threshold = scale_threshold(numerator, denominator)
    if type(prove) is not bool:
        raise ValueError("prove must be a boolean")
    if backend_name not in ("cpu", "cuda"):
        raise ValueError("backend must be cpu or cuda")
    if backend is None:
        from .native_backend import NativeBackend
        backend = NativeBackend(show_progress=progress is not None, backend=backend_name)
    def report(message: str) -> None:
        if progress is not None:
            progress(message)
    model = embedder if embedder is not None else DocumentEmbedder()
    report("Loading model and preparing document embeddings...")
    a, b = normalize(model.embed(text_a, dimension)), normalize(model.embed(text_b, dimension))
    report("Normalizing and quantizing vectors...")
    qa, qb = quantize(a), quantize(b)
    report(f"Generating {backend_name.upper()} keys...")
    context = backend.setup(dimension)
    report("Encrypting vectors" + (" and generating proofs..." if prove else "..."))
    ca, cb = backend.encrypt(context, qa, prove), backend.encrypt(context, qb, prove)
    if prove:
        report("Verifying encryption proofs...")
    if prove and (not backend.verify(context, ca) or not backend.verify(context, cb)):
        raise ValueError("ciphertext proof verification failed")
    report("Evaluating encrypted similarity...")
    encrypted_decision = backend.evaluate(context, ca, cb, threshold)
    report("Decrypting decision and checking reference...")
    decision = backend.decrypt(context, encrypted_decision)
    expected_decision = meets_threshold(qa, qb, numerator, denominator)
    if type(decision) is not bool:
        raise ValueError("backend must decrypt to a boolean decision")
    if decision != expected_decision:
        raise ValueError("decrypted decision does not match the quantized reference")
    floating, quantized = float_similarity(a, b), quantized_similarity(qa, qb)
    report("Complete.")
    return DemoResult(decision, expected_decision, floating, quantized,
                      abs(floating - quantized))
