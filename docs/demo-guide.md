# Demo guide

The demonstration accepts two UTF-8 text file paths, embeds them with `nomic-ai/nomic-embed-text-v1.5` and evaluates a quantized similarity threshold under TFHE. It is a single-process, single-key example for Linux or WSL2. CPU is the default; CUDA is optional.

## Requirements

- Python 3.10–3.14, Rust 1.91.1 or newer, and sufficient disk space for Python, PyTorch and Rust dependencies.
- Network access on the first install and model load. Nomic uses custom model code loaded with `trust_remote_code=True`; its revision is not pinned.
- Time and memory for key generation and homomorphic computation. One observed 64-dimensional CPU evaluation took approximately five minutes; larger dimensions can take substantially longer.

From the repository root, install and run:

```sh
bash scripts/setup.sh
.venv/bin/python -m serenith_demo --help
.venv/bin/python -m serenith_demo examples/abstracts/navier_stokes.txt examples/abstracts/navier_stokes_paraphrase.txt --dimension 64
```

The setup script creates or reuses `.venv`, installs CPU PyTorch when absent, installs the package from `pyproject.toml`, and checks the native extension. Reusing an environment may update packages in it; the script does not delete the environment. The initial Rust release build can take considerable time. Set `SERENITH_PYTHON` to a compatible interpreter if `python3` is not the intended one.

On WSL, keeping the repository in the Linux filesystem avoids slow Windows filesystem operations. If the repository remains on a Windows drive, you can instead place the environment in Linux with:

```sh
CARGO_TARGET_DIR="$HOME/.cache/serenith-fhe-kernel/target" SERENITH_VENV_DIR="$HOME/.venvs/serenith-fhe-kernel" bash scripts/setup.sh
"$HOME/.venvs/serenith-fhe-kernel/bin/python" -m serenith_demo examples/abstracts/navier_stokes.txt examples/abstracts/navier_stokes_paraphrase.txt --dimension 64
```

## Output and options

### CUDA backend

CUDA requires a compatible NVIDIA GPU and driver, a compatible CUDA toolkit with `nvcc`, CMake, libclang and the corresponding build dependencies. The CUDA setup has not been tested on a GPU in this repository; validate it on the target machine before relying on it.

```sh
bash scripts/setup.sh --cuda
.venv/bin/python -m serenith_demo examples/abstracts/navier_stokes.txt examples/abstracts/navier_stokes_paraphrase.txt --dimension 64 --backend cuda
```

When using `SERENITH_VENV_DIR`, invoke the Python executable printed by the setup script instead. A CPU-only extension rejects `--backend cuda` and does not silently fall back. Running setup without `--cuda` again installs a CPU-only extension. GPU and CPU contexts generate independent keys; ZKV2 proofs and GPU results require end-to-end validation on the target hardware. CUDA progress counts completed product batches before the final reduction and comparison.

On a machine with a CUDA-enabled build and compatible GPU, install pytest separately and run the 64D checks:

```sh
.venv/bin/python -m pip install 'pytest>=8,<10'
SERENITH_RUN_CUDA=1 .venv/bin/python -m pytest -q tests/test_cuda.py
SERENITH_RUN_CUDA_PROOF=1 .venv/bin/python -m pytest -q tests/test_cuda.py
```

The first run checks positive and negative decisions; the second checks a proven ciphertext. Neither has been run on a GPU during this change.

### Output

The JSON result contains the decrypted boolean `decision`, a cleartext `reference_decision` computed locally, floating and quantized reference similarities, and their absolute difference. The FHE backend returns only the decision; the reference values come from plaintext data in the demo process.

Use `--dimension 64`, `256` or `768` (default 64). The larger options remain available, but were not tested in this iteration and can take much longer. The default threshold is 4/5; `--threshold-numerator` and `--threshold-denominator` set another rational threshold in [0, 1]. `--prove` requests native ZKV2 ciphertext-construction proofs, verified before evaluation. The proof does not establish document origin or correct embedding, normalization or quantization. The fixed owner metadata is a demo identifier, not a verified identity.

The command prints progress to standard error, leaving the JSON result on standard output. Messages show model loading, key generation, encryption and decryption. During FHE evaluation, the bar counts completed vector coordinates; 100% means the dot product is complete, after which the encrypted threshold comparison still runs.

## Troubleshooting

- **Module not found:** install the package into the same Python environment used to run the command and confirm the native extension was built successfully.
- **Model loading fails:** verify network access, model availability and Python dependencies. Downloaded model code and revisions should be reviewed and pinned for reproducible use.
- **Invalid vector or dimension:** provide supported dimensions and nonempty text producing a finite, nonzero embedding.
- **Evaluation is slow:** begin with 64 dimensions and a release build. CPU execution is not comparable to the PoC's CUDA timings.
- **Decision differs from the reference:** treat this as a failed run; retain the configuration and reproduce it before drawing conclusions.

See `limitations.md` for numerical and security boundaries.
