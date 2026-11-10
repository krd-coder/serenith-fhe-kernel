# Contributing

Thank you for your interest in the public Serenith FHE kernel. Changes should keep the repository focused on the single-key demonstration, its CPU and experimental CUDA backends, numerical references and reproducible tests.

## Development setup

Use Linux or WSL2, a supported Python version and a Rust toolchain. Create a virtual environment and install development dependencies from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

The initial native build and model download may be lengthy. See `docs/demo-guide.md` for the demonstration and `docs/limitations.md` for its scope. The project uses native TFHE-rs; it is not necessary to have CUDA for the public CPU path.

## Changes and checks

Describe the problem, the intended behavior and how the change was checked. Keep numerical scales, supported dimensions and the integer threshold policy consistent across Rust, Python and tests. For changes to cryptographic parameters or the proof statement, document the assumptions and affected trust boundaries.

Run the fast Python tests with `python -m pytest`; run Rust checks and tests with `cargo check --locked` and `cargo test --locked`. Real FHE and ZKV2 tests are intentionally opt-in and expensive; the corresponding environment flags and ignored Rust tests are described in `docs/engineering-practices.md`. Report whether they were run and do not present skipped tests as passing.

For benchmark contributions, provide the command, input seed and digest, hardware, software versions, thread configuration, warmup and repetition counts, individual samples and measurement boundaries. Clearly label single observations and do not substitute estimates for measured results.

## Public and private boundary

This repository contains only material intended for public release. Do not submit private source code, data, credentials, key material or infrastructure details without explicit authorization from their owners. Changes must respect applicable third-party licenses and project ownership decisions. The broader PoC is separately maintained; its full implementation and deployment are outside this repository.
