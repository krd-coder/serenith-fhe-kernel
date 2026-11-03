use rayon::prelude::*;
use tfhe::{FheInt8, FheInt16, FheInt32};

use crate::params;

// TODO: Implement homomorphic L2 norm evaluation with a documented result representation.
// TODO: Implement a parallel dot product with explicit FheInt8-to-FheInt16-to-FheInt32 widening.
// TODO: Compare the encrypted similarity score with threshold T in the quantized domain.
// TODO: Validate vector dimensions and prevent overflow for every supported MRL dimension.
