import numpy as np
import pytest
import serenith_tfhe

from serenith_demo import quantization, reference_math

# TODO: Verify decrypted FHE norm and similarity results exactly match the quantized cleartext reference.
# TODO: Test threshold decisions, boundary values, negative components, and all supported dimensions.
# TODO: Compare quantized results with the float32 reference using a documented error tolerance.
