//! Bounded parallel CUDA evaluation adapted from the Serenith PoC.
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use tfhe::core_crypto::gpu::synchronize_device;
use tfhe::core_crypto::gpu::vec::GpuIndex;
use tfhe::prelude::*;
use tfhe::{CompressedServerKey, CudaServerKey, Device, FheBool, FheInt8, FheInt16, FheInt32};

/// Owns the GPU key and workers. Every worker has its own TFHE thread-local key.
pub struct CudaContext {
    key: CudaServerKey,
    pool: ThreadPool,
}

impl CudaContext {
    pub fn new(client_key: &tfhe::ClientKey) -> Result<Self, String> {
        if tfhe::core_crypto::gpu::get_number_of_gpus() == 0 {
            return Err("No CUDA GPU available".into());
        }
        let key = CompressedServerKey::new(client_key).decompress_to_specific_gpu(GpuIndex::new(0));
        let pool = ThreadPoolBuilder::new().num_threads(4)
            .thread_name(|i| format!("serenith-cuda-{i}"))
            .build().map_err(|error| error.to_string())?;
        pool.broadcast(|_| tfhe::set_server_key(key.clone()));
        Ok(Self { key, pool })
    }

    /// Install the GPU server key on the caller before ciphertext expansion.
    pub fn activate(&self) { tfhe::set_server_key(self.key.clone()); }

    /// Widen i8 inputs to i16 before multiplication, then to i32 before reduction.
    pub fn compare(&self, a: &[FheInt8], b: &[FheInt8], threshold: i32,
                   show_progress: bool) -> Result<FheBool, String> {
        if a.len() != b.len() || a.is_empty() { return Err("invalid ciphertext dimensions".into()); }
        let mut batch_sums = Vec::new();
        let batch_size = self.pool.current_num_threads() * 2;
        if show_progress { eprintln!("CUDA products: {}", crate::similarity_kernel::progress_bar(0, a.len())); }
        for start in (0..a.len()).step_by(batch_size) {
            let end = (start + batch_size).min(a.len());
            let terms: Result<Vec<FheInt32>, String> = self.pool.install(|| (start..end).into_par_iter().map(|i| {
                let left: FheInt16 = a[i].clone().cast_into();
                let right: FheInt16 = b[i].clone().cast_into();
                let product = &left * &right;
                if product.current_device() != Device::CudaGpu {
                    return Err("CUDA product unexpectedly ran on CPU".to_owned());
                }
                Ok(product.cast_into())
            }).collect());
            batch_sums.push(self.sum_values(terms?));
            if show_progress {
                synchronize_device(0);
                eprintln!("CUDA products: {}", crate::similarity_kernel::progress_bar(end, a.len()));
            }
        }
        if show_progress { eprintln!("CUDA: reducing totals and comparing threshold..."); }
        let result = self.sum_values(batch_sums).ge(threshold);
        synchronize_device(0);
        if result.current_device() != Device::CudaGpu {
            return Err("CUDA evaluation unexpectedly returned a CPU result".into());
        }
        Ok(result)
    }

    fn sum_values(&self, mut values: Vec<FheInt32>) -> FheInt32 {
        while values.len() > 1 {
            values = self.pool.install(|| values.par_chunks(2).map(|pair| {
                if pair.len() == 2 { &pair[0] + &pair[1] } else { pair[0].clone() }
            }).collect());
        }
        values.pop().expect("nonempty bounded CUDA batch")
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        self.pool.broadcast(|_| tfhe::clear_gpu_thread_locals());
        tfhe::clear_gpu_thread_locals();
    }
}
