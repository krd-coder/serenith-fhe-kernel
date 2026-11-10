//! Centralized single-key demonstration with CPU and experimental CUDA; no untrusted ciphertext input.
use std::sync::Arc;
use tfhe::prelude::*;
use tfhe::{ClientKey, CompactCiphertextList, CompactPublicKey, FheBool, FheInt8, ProvenCompactCiphertextList, ServerKey};
use tfhe::zk::{CompactPkeCrs, ZkComputeLoad};
use crate::{params, similarity_kernel};

/// Owns secret, public and evaluation keys for one in-memory demonstration.
pub struct CpuBackend {
    client_key: ClientKey,
    public_key: CompactPublicKey,
    server_key: EvaluationKey,
    identity: Arc<()>,
    dimension: usize,
    crs: Option<CompactPkeCrs>,
    show_progress: bool,
}

enum EvaluationKey {
    Cpu(ServerKey),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::CudaContext),
}

/// Compact ciphertext tied to the context that generated it.
pub struct EncryptedVector { list: DemoCiphertext, identity: Arc<()> }
enum DemoCiphertext { Plain(CompactCiphertextList), Proven(ProvenCompactCiphertextList) }
/// Encrypted comparison bit tied to its context.
pub struct EncryptedDecision { value: FheBool, identity: Arc<()> }

// TFHE uses a thread-local server key. Clear the demo key on early return too.
// The calling thread must be dedicated to this operation.
struct ServerKeyGuard;
impl Drop for ServerKeyGuard { fn drop(&mut self) { tfhe::unset_server_key(); } }

impl CpuBackend {
    /// Generate fresh single-key material using the PoC CPU parameters.
    pub fn new(dimension: usize) -> Result<Self, String> {
        params::validate_dimension(dimension).map_err(str::to_owned)?;
        let (client_key, server_key) = tfhe::generate_keys(params::config());
        let public_key = CompactPublicKey::try_new(&client_key).map_err(|e| e.to_string())?;
        Ok(Self { client_key, public_key, server_key: EvaluationKey::Cpu(server_key), identity: Arc::new(()), dimension, crs: None, show_progress: false })
    }

    /// Generate an independent GPU-profile context; CPU keys are not reused.
    #[cfg(feature = "cuda")]
    pub fn new_cuda(dimension: usize) -> Result<Self, String> {
        params::validate_dimension(dimension).map_err(str::to_owned)?;
        if tfhe::core_crypto::gpu::get_number_of_gpus() == 0 {
            return Err("No CUDA GPU available".into());
        }
        let client_key = ClientKey::generate(params::cuda_config());
        let public_key = CompactPublicKey::try_new(&client_key).map_err(|e| e.to_string())?;
        let server_key = crate::cuda::CudaContext::new(&client_key)?;
        Ok(Self { client_key, public_key, server_key: EvaluationKey::Cuda(server_key), identity: Arc::new(()), dimension, crs: None, show_progress: false })
    }

    /// Enable progress reporting for the CPU evaluation on stderr.
    pub fn set_progress(&mut self, enabled: bool) { self.show_progress = enabled; }

    /// Encrypt a quantized vector using a compact public key.
    pub fn encrypt(&self, vector: &[i8]) -> Result<EncryptedVector, String> {
        if vector.len() != self.dimension || vector.contains(&i8::MIN) {
            return Err("vector must match the context dimension and use [-127, 127]".into());
        }
        let mut builder = CompactCiphertextList::builder(&self.public_key);
        for &value in vector { builder.push(value); }
        Ok(EncryptedVector { list: DemoCiphertext::Plain(builder.build_packed()), identity: self.identity.clone() })
    }

    /// Encrypt with a native ZKV2 proof. CRS is generated once per demo context.
    pub fn encrypt_proven(&mut self, vector: &[i8]) -> Result<EncryptedVector, String> {
        if vector.len() != self.dimension || vector.contains(&i8::MIN) {
            return Err("vector must match the context dimension and use [-127, 127]".into());
        }
        if self.crs.is_none() {
            self.crs = Some(CompactPkeCrs::from_config(params::config(), self.dimension * 8)
                .map_err(|e| e.to_string())?);
        }
        let mut builder = CompactCiphertextList::builder(&self.public_key);
        for &value in vector { builder.push(value); }
        let proven = builder.build_with_proof_packed(self.crs.as_ref().unwrap(), &self.proof_metadata(), ZkComputeLoad::Proof)
            .map_err(|e| e.to_string())?;
        Ok(EncryptedVector { list: DemoCiphertext::Proven(proven), identity: self.identity.clone() })
    }

    /// Metadata matches the PoC encoding, with a fixed demo-only owner identifier.
    fn proof_metadata(&self) -> Vec<u8> {
        let mut result = b"serenith/zkv2/i8/scale127/v1".to_vec();
        result.extend_from_slice(&(self.dimension as u16).to_be_bytes());
        result.extend_from_slice(&[1u8; 20]);
        result
    }

    /// Verify the native proof against this context's trusted CRS and public key.
    pub fn verify(&self, ciphertext: &EncryptedVector) -> Result<bool, String> {
        if !Arc::ptr_eq(&self.identity, &ciphertext.identity) {
            return Err("ciphertext must belong to this context".into());
        }
        match (&ciphertext.list, &self.crs) {
            (DemoCiphertext::Proven(list), Some(crs)) => Ok(list.verify(crs, &self.public_key, &self.proof_metadata()).is_valid()),
            (DemoCiphertext::Plain(_), _) => Err("ciphertext does not contain a proof".into()),
            _ => Err("trusted CRS is unavailable".into()),
        }
    }

    /// Evaluate an encrypted dot product against a pre-scaled threshold.
    pub fn evaluate(&self, a: &EncryptedVector, b: &EncryptedVector, threshold: i32)
        -> Result<EncryptedDecision, String>
    {
        if !Arc::ptr_eq(&self.identity, &a.identity) || !Arc::ptr_eq(&self.identity, &b.identity) {
            return Err("ciphertexts must belong to this context".into());
        }
        if !(0..=params::SCALE * params::SCALE).contains(&threshold) {
            return Err("scaled threshold must be between 0 and 16129".into());
        }
        match &self.server_key {
            EvaluationKey::Cpu(key) => tfhe::set_server_key(key.clone()),
            #[cfg(feature = "cuda")]
            EvaluationKey::Cuda(context) => context.activate(),
        }
        let _guard = ServerKeyGuard;
        let expand = |ciphertext: &EncryptedVector| -> Result<Vec<FheInt8>, String> {
            let list = match &ciphertext.list {
                DemoCiphertext::Plain(list) => list.expand().map_err(|e| e.to_string())?,
                DemoCiphertext::Proven(list) => {
                    let crs = self.crs.as_ref().ok_or("trusted CRS is unavailable")?;
                    list.verify_and_expand(crs, &self.public_key, &self.proof_metadata())
                        .map_err(|e| e.to_string())?
                }
            };
            (0..self.dimension).map(|index| {
                list.get(index).map_err(|e| e.to_string())?
                    .ok_or_else(|| "missing ciphertext coordinate".to_owned())
            }).collect()
        };
        if self.show_progress { eprintln!("FHE: expanding ciphertexts..."); }
        let (left, right) = (expand(a)?, expand(b)?);
        if self.show_progress { eprintln!("FHE: calculating encrypted dot product..."); }
        let value = match &self.server_key {
            EvaluationKey::Cpu(_) => {
                let dot = similarity_kernel::dot(&left, &right, self.show_progress).map_err(str::to_owned)?;
                if self.show_progress { eprintln!("FHE: comparing with threshold..."); }
                similarity_kernel::compare(&dot, threshold)
            }
            #[cfg(feature = "cuda")]
            EvaluationKey::Cuda(context) => context.compare(&left, &right, threshold, self.show_progress)?,
        };
        Ok(EncryptedDecision { value, identity: self.identity.clone() })
    }

    /// Decrypt only the decision, with the secret key of this context.
    pub fn decrypt(&self, decision: &EncryptedDecision) -> Result<bool, String> {
        if !Arc::ptr_eq(&self.identity, &decision.identity) {
            return Err("decision must belong to this context".into());
        }
        Ok(decision.value.decrypt(&self.client_key))
    }
}

// Serialized or external ciphertexts are not accepted by this in-memory demo.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unsupported_dimension_before_key_generation() {
        assert!(CpuBackend::new(128).is_err());
    }

    /// Validate progress and a negative decision on real 64-dimensional ciphertexts.
    #[test]
    #[ignore = "slow CPU FHE progress test"]
    fn progress_and_negative_decision_64() {
        let mut backend = CpuBackend::new(64).unwrap();
        backend.set_progress(true);
        let mut a = vec![0i8; 64];
        let mut b = vec![0i8; 64];
        a[0] = 127;
        b[0] = -127;
        let a = backend.encrypt(&a).unwrap();
        let b = backend.encrypt(&b).unwrap();
        let decision = backend.evaluate(&a, &b, params::scaled_threshold(4, 5).unwrap()).unwrap();
        assert!(!backend.decrypt(&decision).unwrap());
    }

    #[test]
    #[ignore = "slow native ZKV2 proof test"]
    fn proof_is_bound_to_context_metadata() {
        let mut backend = CpuBackend::new(64).unwrap();
        let mut vector = vec![0i8; 64];
        vector[0] = 127;
        let ciphertext = backend.encrypt_proven(&vector).unwrap();
        assert!(backend.verify(&ciphertext).unwrap());
        if let DemoCiphertext::Proven(list) = &ciphertext.list {
            let crs = backend.crs.as_ref().unwrap();
            assert!(list.verify(crs, &backend.public_key, b"wrong metadata").is_invalid());
            if let EvaluationKey::Cpu(key) = &backend.server_key { tfhe::set_server_key(key.clone()); }
            let _guard = ServerKeyGuard;
            assert!(list.verify_and_expand(crs, &backend.public_key, b"wrong metadata").is_err());
            let expanded = list.verify_and_expand(crs, &backend.public_key, &backend.proof_metadata()).unwrap();
            let coordinate: FheInt8 = expanded.get(0).unwrap().unwrap();
            let value: i8 = coordinate.decrypt(&backend.client_key);
            assert_eq!(value, 127);
        } else { panic!("proof missing"); }
        assert!(backend.verify(&backend.encrypt(&vector).unwrap()).is_err());
        let other = CpuBackend::new(64).unwrap();
        assert!(other.verify(&ciphertext).is_err());
    }

    /// Real TFHE key generation and 64-dimensional evaluation are expensive on CPU.
    #[test]
    #[ignore = "slow CPU FHE integration test"]
    fn encrypted_decisions_match_integer_reference() {
        let backend = CpuBackend::new(64).unwrap();
        let mut left = vec![0i8; 64];
        let mut right = vec![0i8; 64];
        left[0] = 127;
        right[0] = 127;
        let a = backend.encrypt(&left).unwrap();
        let b = backend.encrypt(&right).unwrap();
        let threshold = params::scaled_threshold(4, 5).unwrap();
        let decision = backend.evaluate(&a, &b, threshold).unwrap();
        assert!(backend.decrypt(&decision).unwrap());
        right[0] = -127;
        let opposite = backend.encrypt(&right).unwrap();
        let decision = backend.evaluate(&a, &opposite, threshold).unwrap();
        assert!(!backend.decrypt(&decision).unwrap());
        assert!(backend.encrypt(&[0; 63]).is_err());
        assert!(backend.evaluate(&a, &b, -1).is_err());
    }
}
