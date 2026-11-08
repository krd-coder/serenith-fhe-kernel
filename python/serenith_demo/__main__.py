"""Run the single-key demonstration with two text inputs."""
import argparse
import json
import sys
from dataclasses import asdict

from .pipeline import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Serenith single-key demonstration (CPU by default; CUDA experimental)")
    parser.add_argument("text_a")
    parser.add_argument("text_b")
    parser.add_argument("--dimension", type=int, choices=(64, 256, 768), default=64,
                        help="MRL vector dimension (default: 64; larger dimensions take longer)")
    parser.add_argument("--threshold-numerator", type=int, default=4)
    parser.add_argument("--threshold-denominator", type=int, default=5)
    parser.add_argument("--prove", action="store_true", help="create and verify native ZKV2 encryption proofs")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu",
                        help="evaluation backend (default: cpu; cuda needs a CUDA-enabled build and NVIDIA GPU)")
    args = parser.parse_args()
    result = run_demo(args.text_a, args.text_b, args.dimension,
                      args.threshold_numerator, args.threshold_denominator, prove=args.prove,
                      progress=lambda message: print(message, file=sys.stderr, flush=True),
                      backend_name=args.backend)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
