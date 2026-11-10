"""Measure the single-key CPU demo using reproducible synthetic embeddings.

Run in an installed release build, e.g. python benchmarks/bench_mrl_dimensions.py
--dimensions 64 --warmup 0 --repetitions 1 --output results.json. Evaluation
includes ciphertext expansion. Setup is measured once per dimension; encryption
times both vectors. Model inference and proof generation are not included.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
from serenith_demo.native_backend import NativeBackend
from serenith_demo.quantization import SUPPORTED_DIMENSIONS, normalize, quantize, scale_threshold
from serenith_demo.reference_math import float_similarity, quantized_similarity, meets_threshold


def stats(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values),
            "mean": statistics.mean(values), "samples": values}


def hardware_details():
    """Read Linux/WSL model and memory where available; leave unavailable fields null."""
    cpu = platform.processor() or None
    memory = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("MemTotal:"):
                    memory = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    return cpu, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", choices=SUPPORTED_DIMENSIONS, default=[64])
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", required=True, help="new output JSON path")
    parser.add_argument("--build-profile", choices=("release", "debug", "unknown"), default="unknown",
                        help="record the build profile of the installed native extension")
    args = parser.parse_args()
    if args.seed < 0 or args.warmup < 0 or args.repetitions < 1 or len(set(args.dimensions)) != len(args.dimensions):
        parser.error("seed and warmup must be nonnegative, repetitions positive, dimensions unique")
    if os.path.exists(args.output):
        parser.error("output already exists; choose a new path")

    cpu, memory = hardware_details()
    result = {"configuration": {"dimensions": args.dimensions, "seed": args.seed,
        "warmup": args.warmup, "repetitions": args.repetitions, "threshold": [4, 5],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "build_profile": args.build_profile,
        "tfhe_version": "1.8.2", "python": sys.version, "numpy": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": cpu, "memory_bytes": memory, "logical_cpus": os.cpu_count(),
        "rayon_num_threads": os.getenv("RAYON_NUM_THREADS"),
        "scope": "CPU single-key; evaluate includes expansion; no model inference or ZK proofs"},
        "results": []}
    backend = NativeBackend()
    for dimension in args.dimensions:
        rng = np.random.default_rng(args.seed + dimension)
        a, b = normalize(rng.normal(size=dimension)), normalize(rng.normal(size=dimension))
        qa, qb = quantize(a), quantize(b)
        digest = hashlib.sha256(qa.tobytes() + qb.tobytes()).hexdigest()
        expected = meets_threshold(qa, qb, 4, 5)
        threshold = scale_threshold(4, 5)
        start = perf_counter(); ctx = backend.setup(dimension); setup = perf_counter() - start
        samples = {"encrypt_two_seconds": [], "evaluate_seconds": [], "decrypt_seconds": []}
        for index in range(args.warmup + args.repetitions):
            start = perf_counter(); ca = backend.encrypt(ctx, qa, False); cb = backend.encrypt(ctx, qb, False)
            encryption = perf_counter() - start
            start = perf_counter(); ct = backend.evaluate(ctx, ca, cb, threshold); evaluation = perf_counter() - start
            start = perf_counter(); decision = backend.decrypt(ctx, ct); decryption = perf_counter() - start
            if type(decision) is not bool or decision != expected:
                raise RuntimeError("FHE decision differs from the quantized reference")
            if index >= args.warmup:
                for key, value in zip(samples, (encryption, evaluation, decryption)):
                    samples[key].append(value)
        floating, integer = float_similarity(a, b), quantized_similarity(qa, qb)
        result["results"].append({"dimension": dimension, "quantized_inputs_sha256": digest,
            "setup_seconds": setup, "timings": {k: stats(v) for k, v in samples.items()},
            "float_similarity": floating, "quantized_similarity": integer,
            "absolute_quantization_error": abs(floating - integer), "decision": expected})
    with open(args.output, "x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
