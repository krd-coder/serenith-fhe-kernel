"""CLI dimension selection and progress output."""
import sys
from dataclasses import dataclass
import pytest

from serenith_demo import __main__ as cli


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_cli_forwards_dimension_and_prints_progress(monkeypatch, capsys, tmp_path, backend):
    seen = {}

    @dataclass
    class Result:
        decision: bool = True

    def fake_run(a, b, dimension, numerator, denominator, *, prove, progress, backend_name):
        seen["texts"] = (a, b)
        seen["dimension"] = dimension
        seen["backend"] = backend_name
        progress("Evaluating encrypted similarity...")
        return Result()

    monkeypatch.setattr(cli, "run_demo", fake_run)
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("First document", encoding="utf-8")
    b.write_text("Second document", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["serenith_demo", str(a), str(b), "--dimension", "64", "--backend", backend])
    cli.main()
    output = capsys.readouterr()
    assert seen["texts"] == ("First document", "Second document")
    assert seen["dimension"] == 64
    assert seen["backend"] == backend
    assert "Evaluating encrypted similarity" in output.err
    assert '"decision": true' in output.out


@pytest.mark.parametrize("contents", [None, b"", b" \n", b"\xff"])
def test_cli_rejects_missing_empty_or_non_utf8_file(monkeypatch, capsys, tmp_path, contents):
    path = tmp_path / "input.txt"
    if contents is not None:
        path.write_bytes(contents)
    monkeypatch.setattr(sys, "argv", ["serenith_demo", str(path), str(path)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert str(path) in capsys.readouterr().err
