#!/usr/bin/env bash
# Set up the single-key demonstration on Linux or WSL (CPU by default; CUDA optional).
set -euo pipefail

cuda_enabled=false
case "${1:-}" in
    "") ;;
    --cuda) cuda_enabled=true ;;
    -h|--help) echo "Usage: bash scripts/setup.sh [--cuda]"; exit 0 ;;
    *) echo "Usage: bash scripts/setup.sh [--cuda]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Usage: bash scripts/setup.sh [--cuda]" >&2
    exit 2
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
venv_dir="${SERENITH_VENV_DIR:-$project_dir/.venv}"
if [[ "$venv_dir" != /* ]]; then
    venv_dir="$project_dir/$venv_dir"
fi
python_bin="${SERENITH_PYTHON:-python3}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This installer supports Linux or WSL. On Windows, run it inside WSL." >&2
    exit 1
fi
for tool in "$python_bin" cargo rustc; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Missing required tool: $tool. Install it before running this script." >&2
        exit 1
    fi
done
if "$cuda_enabled"; then
    for tool in nvcc cmake; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "Missing CUDA build tool: $tool. Install the CUDA toolkit and CMake first." >&2
            exit 1
        fi
    done
fi
if ! "$python_bin" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] < (3, 15)))'; then
    echo "Python 3.10–3.14 is required; set SERENITH_PYTHON to a compatible interpreter." >&2
    exit 1
fi
rust_version="$(rustc --version)"
if ! "$python_bin" -c 'import re,sys; m=re.match(r"rustc (\d+)\.(\d+)\.(\d+)", sys.argv[1]); sys.exit(not (m and tuple(map(int, m.groups())) >= (1, 91, 1)))' "$rust_version"; then
    echo "Rust 1.91.1 or newer is required by TFHE-rs 1.8.2; found: $rust_version" >&2
    exit 1
fi

cd "$project_dir"
if [[ -L "$venv_dir" ]]; then
    echo "Refusing to use a symbolic link at $venv_dir; use a project-owned virtual environment." >&2
    exit 1
elif [[ -e "$venv_dir" ]]; then
    if [[ ! -f "$venv_dir/pyvenv.cfg" || ! -x "$venv_dir/bin/python" ]]; then
        echo "$venv_dir exists but is not a usable Python virtual environment; it was not changed." >&2
        exit 1
    fi
    echo "[1/4] Reusing existing virtual environment: $venv_dir"
else
    echo "[1/4] Creating virtual environment: $venv_dir"
    "$python_bin" -m venv "$venv_dir"
fi
venv_python="$venv_dir/bin/python"
if ! "$venv_python" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] < (3, 15)))'; then
    echo "Existing virtual environment uses an unsupported Python version; it was not changed." >&2
    exit 1
fi
if ! "$venv_python" -m pip --version >/dev/null 2>&1; then
    echo "pip is missing from the virtual environment; install it before continuing." >&2
    exit 1
fi

echo "[2/4] Checking PyTorch (downloads the CPU build if absent)..."
if "$venv_python" -c 'import torch' >/dev/null 2>&1; then
    echo "Using the PyTorch already installed in this virtual environment."
else
    "$venv_python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.2,<3'
fi

echo "[3/4] Installing the package and building the Rust extension. The first build may take a while..."
if "$cuda_enabled"; then
    "$venv_python" -m pip install 'maturin>=1.8,<2'
    PATH="$venv_dir/bin:$PATH" VIRTUAL_ENV="$venv_dir" "$venv_dir/bin/maturin" develop --release --locked --features cuda
else
    "$venv_python" -m pip install .
fi

echo "[4/4] Checking the Python dependencies and native extension..."
"$venv_python" -c 'import einops, sentence_transformers; from serenith_demo.serenith_tfhe import validate_dimension; validate_dimension(64)'

echo "Setup complete. Run the 64-dimensional demo with:"
printf '  %q -m serenith_demo examples/abstracts/navier_stokes.txt examples/abstracts/navier_stokes_paraphrase.txt --dimension 64\n' "$venv_python"
echo 'Add --prove to generate and verify ZKV2 encryption proofs.'
if "$cuda_enabled"; then
    echo 'Add --backend cuda to use the CUDA-enabled extension and an NVIDIA GPU.'
fi
echo 'The first demo run downloads the Nomic model and its custom model code.'
