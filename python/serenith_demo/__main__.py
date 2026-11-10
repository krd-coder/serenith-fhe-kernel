"""Run the single-key demonstration with two UTF-8 text files."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .pipeline import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Serenith single-key demonstration (CPU by default; CUDA experimental)")
    parser.add_argument("file_a", type=Path, help="first UTF-8 text file")
    parser.add_argument("file_b", type=Path, help="second UTF-8 text file")
    parser.add_argument("--dimension", type=int, choices=(64, 256, 768), default=64,
                        help="MRL vector dimension (default: 64; larger dimensions take longer)")
    parser.add_argument("--threshold-numerator", type=int, default=4)
    parser.add_argument("--threshold-denominator", type=int, default=5)
    parser.add_argument("--prove", action="store_true", help="create and verify native ZKV2 encryption proofs")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu",
                        help="evaluation backend (default: cpu; cuda needs a CUDA-enabled build and NVIDIA GPU)")
    args = parser.parse_args()
    texts = []
    for path in (args.file_a, args.file_b):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            parser.error(f"cannot read UTF-8 text file {path}: {exc}")
        if not text.strip():
            parser.error(f"text file is empty: {path}")
        texts.append(text)
    result = run_demo(texts[0], texts[1], args.dimension,
                      args.threshold_numerator, args.threshold_denominator, prove=args.prove,
                      progress=lambda message: print(message, file=sys.stderr, flush=True),
                      backend_name=args.backend)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
