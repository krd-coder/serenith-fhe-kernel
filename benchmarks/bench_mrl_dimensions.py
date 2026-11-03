from time import perf_counter

import numpy as np

from serenith_demo import embedder, pipeline, quantization, reference_math

# TODO: Benchmark FHE execution time and quantization error for MRL dimensions 64, 128, and 256.
# TODO: Separate setup, encryption, evaluation, and decryption timings and define warmup and repetition counts.
# TODO: Record reproducible inputs, seeds, CPU details, software versions, and summary statistics.
