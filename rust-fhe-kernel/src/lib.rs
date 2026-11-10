//! Serenith encrypted-similarity kernel. The Python bridge is under development.
pub mod encryption;
mod params;
mod similarity_kernel;
#[cfg(feature = "cuda")]
mod cuda;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Opaque, in-memory single-key demo context for CPU or experimental CUDA.
#[pyclass(unsendable)]
struct CpuContext { inner: encryption::CpuBackend }

/// An opaque compact ciphertext tied to its originating context.
#[pyclass(unsendable)]
struct Ciphertext { inner: encryption::EncryptedVector }

/// An opaque encrypted matching bit.
#[pyclass(unsendable)]
struct Decision { inner: encryption::EncryptedDecision }

#[pymethods]
impl CpuContext {
    #[new]
    #[pyo3(signature = (dimension, backend="cpu"))]
    fn new(dimension: usize, backend: &str) -> PyResult<Self> {
        let inner = match backend {
            "cpu" => encryption::CpuBackend::new(dimension),
            #[cfg(feature = "cuda")]
            "cuda" => encryption::CpuBackend::new_cuda(dimension),
            #[cfg(not(feature = "cuda"))]
            "cuda" => Err("CUDA support is not enabled in this build".to_owned()),
            _ => Err("backend must be cpu or cuda".to_owned()),
        };
        Ok(Self { inner: inner.map_err(PyValueError::new_err)? })
    }

    #[pyo3(signature = (vector, prove=false))]
    fn encrypt(&mut self, vector: Vec<i8>, prove: bool) -> PyResult<Ciphertext> {
        let inner = if prove { self.inner.encrypt_proven(&vector) } else { self.inner.encrypt(&vector) };
        Ok(Ciphertext { inner: inner.map_err(PyValueError::new_err)? })
    }

    fn verify(&self, ciphertext: &Ciphertext) -> PyResult<bool> {
        self.inner.verify(&ciphertext.inner).map_err(PyValueError::new_err)
    }

    fn set_progress(&mut self, enabled: bool) { self.inner.set_progress(enabled); }

    fn evaluate(&self, a: &Ciphertext, b: &Ciphertext, threshold: i32) -> PyResult<Decision> {
        Ok(Decision { inner: self.inner.evaluate(&a.inner, &b.inner, threshold).map_err(PyValueError::new_err)? })
    }

    fn decrypt(&self, decision: &Decision) -> PyResult<bool> {
        self.inner.decrypt(&decision.inner).map_err(PyValueError::new_err)
    }
}

/// Expose the same rational threshold conversion as the PoC.
#[pyfunction]
fn scaled_threshold(numerator: u32, denominator: u32) -> PyResult<i32> {
    params::scaled_threshold(numerator, denominator).map_err(PyValueError::new_err)
}

/// Validate a dimension before the FHE backend processes a vector.
#[pyfunction]
fn validate_dimension(dimension: usize) -> PyResult<()> {
    params::validate_dimension(dimension).map_err(PyValueError::new_err)
}

/// Whether this extension was compiled with the CUDA feature.
#[pyfunction]
fn cuda_compiled() -> bool { cfg!(feature = "cuda") }

/// Register the extension; additional FHE APIs are added with their validated bridge.
#[pymodule]
fn serenith_tfhe(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CpuContext>()?;
    module.add_class::<Ciphertext>()?;
    module.add_class::<Decision>()?;
    module.add_function(wrap_pyfunction!(scaled_threshold, module)?)?;
    module.add_function(wrap_pyfunction!(validate_dimension, module)?)?;
    module.add_function(wrap_pyfunction!(cuda_compiled, module)?)?;
    Ok(())
}
