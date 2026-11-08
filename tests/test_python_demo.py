"""Offline tests for the Python preprocessing and backend integration contract."""

import unittest
import numpy as np

from serenith_demo.embedder import DocumentEmbedder
from serenith_demo.pipeline import run_demo
from serenith_demo.quantization import normalize, quantize, scale_threshold
from serenith_demo.reference_math import integer_dot, integer_squared_norm, float_similarity


class Model:
    def encode(self, text, convert_to_numpy):
        assert text.startswith("search_document: ") and convert_to_numpy
        v = np.zeros(768, dtype=np.float32)
        v[0] = 1 if "first" in text else -1
        return v


class Backend:
    def setup(self, dimension): return dimension
    def encrypt(self, context, vector, prove): return vector
    def verify(self, context, ciphertext): return True
    def evaluate(self, context, a, b, threshold):
        dot = integer_dot(a, b)
        return dot >= threshold
    def decrypt(self, context, result): return result


class PythonDemoTests(unittest.TestCase):
    def test_dimensions_and_opposite_vectors(self):
        for dim in (64,):
            x = np.zeros(dim); x[0] = 1
            q = quantize(normalize(x))
            self.assertEqual(integer_dot(q, -q), -16129)
            self.assertEqual(integer_squared_norm(q), 16129)
            self.assertAlmostEqual(float_similarity(x, -x), -1)

    def test_threshold_rounds_up(self):
        self.assertEqual(scale_threshold(4, 5), 12904)
        self.assertEqual(scale_threshold(0, 1), 0)
        self.assertEqual(scale_threshold(1, 1), 16129)
        for args in ((1, 0), (2, 1), (True, 1), (1, 1000001)):
            with self.assertRaises(ValueError): scale_threshold(*args)

    def test_invalid_vectors(self):
        for v in (np.zeros(64), np.full(64, np.nan), np.ones(128), np.ones((64, 1)), np.ones(64, dtype=complex)):
            with self.assertRaises(ValueError): normalize(v)
        with self.assertRaises(ValueError): quantize(np.ones(64))

    def test_round_to_even(self):
        x = np.zeros(64); x[0] = 0.5 / 127; x[1] = 1.5 / 127
        x[2] = np.sqrt(1 - x[0] ** 2 - x[1] ** 2)
        q = quantize(x)
        self.assertEqual((q[0], q[1]), (0, 2))

    def test_pipeline_and_failed_proof(self):
        model = DocumentEmbedder(Model())
        result = run_demo("first", "second", 64, 4, 5, Backend(), embedder=model, prove=True)
        self.assertFalse(result.decision)
        self.assertFalse(result.reference_decision)
        class BadProof(Backend):
            def verify(self, context, ciphertext): return False
            def evaluate(self, *args): raise AssertionError("must not evaluate")
        with self.assertRaises(ValueError):
            run_demo("first", "second", 64, 4, 5, BadProof(), embedder=model, prove=True)

    def test_progress_reports_real_stages(self):
        messages = []
        run_demo("first", "second", 64, 4, 5, Backend(),
                 embedder=DocumentEmbedder(Model()), progress=messages.append)
        self.assertTrue(any("embeddings" in item for item in messages))
        self.assertTrue(any("keys" in item for item in messages))
        self.assertTrue(any("encrypted similarity" in item for item in messages))
        self.assertEqual(messages[-1], "Complete.")

    def test_layer_normalization_precedes_prefix(self):
        class FullModel:
            def encode(self, text, convert_to_numpy):
                return np.arange(768, dtype=np.float32)
        actual = DocumentEmbedder(FullModel()).embed("document", 64)
        full = np.arange(768, dtype=np.float64)
        centered = full - full.mean()
        expected = (centered / np.sqrt(centered.var() + 1e-5))[:64]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertLess(actual[-1], 0)  # Prefix-only centering would change its sign.

    def test_invalid_decision(self):
        class InvalidBackend(Backend):
            def decrypt(self, context, result): return 0
        with self.assertRaises(ValueError):
            run_demo("first", "second", 64, 4, 5, InvalidBackend(), embedder=DocumentEmbedder(Model()))


if __name__ == "__main__": unittest.main()
