# Engineering practices

## Python and Rust interfaces

Python modules use type annotations for public interfaces and NumPy arrays for numeric vectors. Supported dimensions and quantization constants are defined explicitly. The Rust kernel uses opaque context, ciphertext and decision types so a caller cannot combine objects from different key contexts through the demo API.

Python validates embedding and quantization inputs before native calls. Rust validates dimensions, ranges and context ownership independently. Invalid values and context mismatches are exposed as `ValueError`; invalid Python argument types can produce `TypeError` or `OverflowError` during PyO3 conversion. Failures from cryptographic operations are currently also mapped to `ValueError` and must not be interpreted as proof of input validity.

## Numerical reproducibility

Normalized vectors use scale 127 and nearest, ties-to-even rounding. Integer thresholds are rounded upward. Accumulators widen before summation. Compare encrypted decisions with the quantized integer reference separately from the floating-point approximation.

The benchmark generates inputs with a recorded NumPy seed and stores their quantized SHA-256 digest, versions, timings and environment details. Encryption, key generation and proofs use cryptographic randomness and are not made deterministic by that seed. Record a separate lockfile and model revision for fully reproducible software and model inputs.

## Execution and tests

The current CPU dot-product loop is sequential; TFHE may use worker threads internally. The optional CUDA path uses a bounded worker pool and pairwise reductions, following the PoC. Benchmarks record `RAYON_NUM_THREADS` when set. The backend installs a thread-local TFHE server key during evaluation and clears it afterwards; use a dedicated calling thread, as it does not restore an existing key on that thread.

Fast Python tests cover validation, numerical reference behavior and benchmark accounting. Real CPU FHE and proof tests are explicitly enabled via `SERENITH_RUN_FHE=1` and `SERENITH_RUN_ZK=1`; the 64D GPU decision and proof tests require `SERENITH_RUN_CUDA=1` and `SERENITH_RUN_CUDA_PROOF=1` respectively, along with compatible hardware. Benchmark results must state warmup, repetition counts and measurement boundaries.
