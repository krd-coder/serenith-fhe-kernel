"""CLI dimension selection and progress output."""
import sys
from dataclasses import dataclass
import pytest

from serenith_demo import __main__ as cli


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_cli_forwards_dimension_and_prints_progress(monkeypatch, capsys, backend):
    seen = {}

    @dataclass
    class Result:
        decision: bool = True

    def fake_run(a, b, dimension, numerator, denominator, *, prove, progress, backend_name):
        seen["dimension"] = dimension
        seen["backend"] = backend_name
        progress("Evaluating encrypted similarity...")
        return Result()

    monkeypatch.setattr(cli, "run_demo", fake_run)
    monkeypatch.setattr(sys, "argv", ["serenith_demo", "a", "b", "--dimension", "64", "--backend", backend])
    cli.main()
    output = capsys.readouterr()
    assert seen["dimension"] == 64
    assert seen["backend"] == backend
    assert "Evaluating encrypted similarity" in output.err
    assert '"decision": true' in output.out
