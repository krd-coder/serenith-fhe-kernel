"""Validation tests at the Python–Rust boundary."""
import pytest

native = pytest.importorskip("serenith_demo.serenith_tfhe")


def test_import_and_threshold():
    assert native.scaled_threshold(4, 5) == 12904
    assert native.validate_dimension(64) is None


@pytest.mark.parametrize("dimension", [0, 128, 769])
def test_invalid_dimensions_raise_value_error(dimension):
    with pytest.raises(ValueError):
        native.CpuContext(dimension)


def test_invalid_threshold_raises_value_error():
    with pytest.raises(ValueError):
        native.scaled_threshold(1, 0)


def test_native_ciphertexts_cannot_be_constructed_by_caller():
    with pytest.raises(TypeError):
        native.Ciphertext()
    with pytest.raises(TypeError):
        native.Decision()

# Cross-context and malformed ciphertext checks require actual key setup; see
# the opt-in CPU integration tests. External serialized ciphertexts are not accepted.
