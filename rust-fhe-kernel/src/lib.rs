use pyo3::prelude::*;

mod encryption;
mod params;
mod similarity_kernel;

// TODO: Define the serenith_tfhe PyO3 module and export its Python-facing functions.
// TODO: Map Rust validation errors to Python ValueError and document other exception mappings.
// TODO: Define ownership and lifetimes for keys, ciphertexts, and evaluation results crossing the bridge.
