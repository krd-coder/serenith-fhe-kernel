"""Explicitly run after building the native extension; real FHE on CPU is slow."""
import os
import unittest
import numpy as np

from serenith_demo.embedder import DocumentEmbedder
from serenith_demo.native_backend import NativeBackend
from serenith_demo.pipeline import run_demo

try:
    from serenith_demo import serenith_tfhe
except ImportError:
    serenith_tfhe = None


class FixedModel:
    def encode(self, text, convert_to_numpy):
        return np.arange(768, dtype=np.float32)


class NativeIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("SERENITH_RUN_FHE") == "1", "slow CPU FHE test")
    def test_native_decision_matches_cleartext(self):
        result = run_demo("first text", "second text", 64, 4, 5,
                          NativeBackend(), embedder=DocumentEmbedder(FixedModel()))
        self.assertTrue(result.decision)
        self.assertEqual(result.decision, result.reference_decision)

    def test_invalid_dimension_rejected_before_key_setup(self):
        if serenith_tfhe is None:
            self.skipTest("native extension is not installed")
        with self.assertRaises(ValueError):
            serenith_tfhe.CpuContext(128)

    @unittest.skipUnless(os.environ.get("SERENITH_RUN_ZK") == "1", "slow ZKV2 proof test")
    def test_proven_ciphertext_verifies(self):
        backend = NativeBackend()
        context = backend.setup(64)
        vector = np.zeros(64, dtype=np.int8)
        vector[0] = 127
        ciphertext = backend.encrypt(context, vector, True)
        self.assertTrue(backend.verify(context, ciphertext))
        plain = backend.encrypt(context, vector, False)
        with self.assertRaises(ValueError):
            backend.verify(context, plain)


if __name__ == "__main__":
    unittest.main()
