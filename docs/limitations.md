# Limitations

This repository demonstrates local single-key CPU computation and an experimental CUDA path, not the complete Serenith PoC or a decentralized key-management network. The CUDA path has not been validated on an NVIDIA GPU in this repository. The demo keeps the secret key in the same process as the evaluation key. Text and embeddings are visible to the environment that prepares them, and reference scores are calculated locally.

The supported vector dimensions are 64, 256 and 768. Normalized coordinates are rounded to nearest with ties to even, using scale 127 and signed i8 values in [-127, 127]. Products are widened before accumulation in signed i32. The supported rational threshold is in [0, 1], with denominator at most 1,000,000; the integer comparison threshold is rounded upward.

Quantization approximates floating-point similarity. For two normalized d-dimensional vectors, nearest rounding gives an absolute dot-product error bound of sqrt(d)/127 + d/(4*127²), apart from floating-point error. The encrypted comparison follows the quantized integer reference, not the original floating-point score. Scores near a threshold may therefore produce different decisions.

CPU execution is computationally expensive. One local 64-dimensional run on an Intel Core i7-1260P under WSL2 measured approximately 309 seconds for ciphertext expansion and evaluation; this was one observation, not a representative benchmark. Results for other dimensions and environments must be measured separately.

Optional proofs establish the validity of ciphertext construction under the demo's key and CRS. They do not establish document origin, correct model inference, normalization or quantization. This local demo uses a fixed demo-only owner identifier in proof metadata; it does not authenticate a real user or implement on-chain ownership. It does not accept serialized external ciphertexts.

The Nomic model is loaded with custom remote code. Model and code revisions are not currently pinned; first use downloads them and therefore requires network access. Fixing revisions and validating their provenance are necessary for reproducible runs.
