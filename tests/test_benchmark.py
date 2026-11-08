"""Exercise benchmark accounting without pretending to measure FHE performance."""
import importlib.util
import json
from pathlib import Path
import sys

from serenith_demo.reference_math import integer_dot


def test_benchmark_records_reproducible_inputs_and_timing(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_mrl_dimensions.py"
    spec = importlib.util.spec_from_file_location("benchmark_module", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    class FakeBackend:
        def setup(self, dimension): return dimension
        def encrypt(self, context, vector, prove): return vector
        def evaluate(self, context, a, b, threshold): return integer_dot(a, b) >= threshold
        def decrypt(self, context, result): return result

    monkeypatch.setattr(benchmark, "NativeBackend", FakeBackend)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--dimensions", "64", "--warmup", "1",
                                     "--repetitions", "2", "--seed", "42", "--output", str(output)])
    benchmark.main()
    data = json.loads(output.read_text())
    assert data["configuration"]["seed"] == 42
    result = data["results"][0]
    assert result["dimension"] == 64
    assert len(result["quantized_inputs_sha256"]) == 64
    assert len(result["timings"]["evaluate_seconds"]["samples"]) == 2
    assert result["absolute_quantization_error"] >= 0
