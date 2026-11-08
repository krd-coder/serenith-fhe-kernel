# Serenith FHE Kernel

Serenith FHE Kernel is a public repository for exploring privacy-preserving similarity matching between document representations. It presents selected ideas from Serenith’s broader, separately maintained proof of concept through an independently inspectable FHE kernel and a linear demonstration.

The broader Serenith architecture combines encrypted computation with decentralized coordination. It explores how homomorphic encryption can support matching on protected document representations, while smart contracts and cryptographic proofs provide a foundation for verifiable protocol interactions. This repository focuses on the FHE kernel within that design, offering accessible examples of project ideas and the technical capabilities of the team.

**Status:** the demonstration connects the Python pipeline to the Rust FHE kernel, with optional native ZKV2 ciphertext-construction proofs. The CPU 64-dimensional path has been exercised. Larger dimensions have not been validated in this iteration.

The default demonstration uses CPU. An optional CUDA path adapts the broader PoC's GPU parameters and bounded parallel evaluation; it requires a CUDA-enabled build and a compatible NVIDIA GPU.

## Workflow

The demonstration brings together the following stages for two text inputs:

1. Produce embeddings with an MRL-compatible model, apply layer normalization to each full embedding and select a supported prefix.
2. Normalize each prefix using its L2 norm and quantize it to signed 8-bit integers with scale `S = 127`.
3. Encrypt the vectors and evaluate their similarity using TFHE operations in Rust.
4. Decrypt the matching decision in a single-key demo and compare it with a cleartext reference decision.

The Python layer orchestrates the demonstration and reference calculations. A Rust extension exposed through PyO3 performs the FHE operations. Tests check correctness against the quantized reference, and benchmarks are intended to report measured performance and quantization error.

## Public and private scope

| Area | Public kernel repository | Broader proof of concept |
| --- | --- | --- |
| Purpose | Demonstrate selected numerical and FHE ideas. | Explore a larger part of the proposed Serenith system. |
| Embeddings and computation | Linear Python demo and Rust TFHE kernel. | Contains a broader client and encrypted-computation workflow. |
| Coordination and result handling | Outside the scope of the single-key demo. | Explored by additional protocol components; integrations are at different stages of development. |
| Distribution | Open for public inspection. | Separately maintained; not fully released in this repository. |

The broader proof of concept brings these ideas together with additional protocol components.

## Repository layout

- `python/serenith_demo/` — embedding, quantization, reference math and pipeline components.
- `rust-fhe-kernel/` — TFHE operations and Python extension.
- `tests/` — tests for quantization, FHE correctness and the Rust–Python interface.
- `benchmarks/` — measurements and their results.
- `docs/` — architecture and supporting documentation.
- `examples/abstracts/` — four short text samples for similarity comparisons.

## Quick start

On Linux or WSL, with Python 3.10–3.14 and Rust 1.91.1 or newer installed:

```sh
bash scripts/setup.sh
.venv/bin/python -m serenith_demo --help
.venv/bin/python -m serenith_demo examples/abstracts/navier_stokes.txt examples/abstracts/navier_stokes_paraphrase.txt --dimension 64
```

The two positional arguments are paths to UTF-8 text files. The sample pair above contains two descriptions of the same Navier–Stokes research and is intended to return `decision: true` at the default threshold. A second pair covers the quasi-Riemann hypothesis:

```sh
.venv/bin/python -m serenith_demo examples/abstracts/riemann.txt examples/abstracts/riemann_paraphrase.txt --dimension 64
```

To compare different topics, use `examples/abstracts/navier_stokes.txt` and `examples/abstracts/riemann.txt`. With the default 64-dimensional configuration and threshold of 0.8, the cleartext reference produced `true` for each matching pair and `false` across these topics in a local model run. The encrypted result should match the reference.

The four files are short, original summaries written for this demonstration, not verbatim abstracts. The Navier–Stokes samples refer to [OpenAI’s research announcement](https://openai.com/index/navier-stokes-solution/); the Riemann samples refer to [OpenAI’s quasi-Riemann manuscript](https://github.com/openai/math/tree/main/preprints/The-Quasi-Riemann-Hypothesis-September-30-2026). Results depend on the embedding model, dimension and threshold; `decision` is compared with `reference_decision`.

Setup downloads the Python dependencies and builds the Rust extension. The first demo run downloads the Nomic model and its custom model code. The single-key demo performs FHE computation on CPU, which can take several minutes. See [the demo guide](docs/demo-guide.md) for WSL environment placement and troubleshooting.

Select a vector size with `--dimension 64`, `--dimension 256` or `--dimension 768`; the default is 64. Progress is shown on standard error: the coordinate bar covers the encrypted dot product, followed by the final threshold comparison. JSON results remain on standard output. Add `--prove` to generate and verify native ZKV2 encryption proofs; these do not establish document provenance or correct preprocessing.

For the GPU path, install with `bash scripts/setup.sh --cuda` and add `--backend cuda` when running the demo. CUDA and CPU contexts generate separate keys. See the demo guide for GPU prerequisites.

## Security and limitations

FHE evaluation does not protect plaintext from the environment that prepares the inputs. A proof associated with ciphertext construction must not be treated as proof of document provenance or correct preprocessing unless those claims are separately established. The demonstration uses a single key holder and does not implement the full protocol or distributed key management.

See [Architecture](docs/architecture.md) for the data flow and trust boundaries, [limitations](docs/limitations.md) for the demo scope, and [security](SECURITY.md) and [contributing](CONTRIBUTING.md) for project practices. The repository’s license has not yet been selected; see [LICENSE](LICENSE).
