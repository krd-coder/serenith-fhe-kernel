//! Encrypted dot product, squared norm and comparison for the single-key demo.
use tfhe::prelude::*;
use tfhe::{FheBool, FheInt8, FheInt16, FheInt32};
use std::io::{self, IsTerminal, Write};

use crate::params::validate_dimension;

pub(crate) fn progress_bar(completed: usize, total: usize) -> String {
    let filled = completed * 20 / total;
    let percent = completed * 100 / total;
    format!("[{}{}] {}/{} ({}%)", "#".repeat(filled), "-".repeat(20 - filled), completed, total, percent)
}

/// Accumulate products in i32, widening each i8 operand before multiplication.
/// The worst case, 768 * 128² = 12,582,912, fits in i32.
// TODO: Implement and benchmark bounded parallel products and tree reduction;
// worker threads must receive the correct TFHE server key.
pub fn dot(a: &[FheInt8], b: &[FheInt8], show_progress: bool) -> Result<FheInt32, &'static str> {
    validate_dimension(a.len())?;
    if a.len() != b.len() { return Err("vector dimensions must match"); }
    let mut result = FheInt32::encrypt_trivial(0i32);
    let terminal = io::stderr().is_terminal();
    let mut last_percent = 0;
    if show_progress {
        if terminal { eprint!("\rFHE: {}", progress_bar(0, a.len())); }
        else { eprintln!("FHE: {}", progress_bar(0, a.len())); }
        let _ = io::stderr().flush();
    }
    for (index, (left, right)) in a.iter().zip(b).enumerate() {
        let left: FheInt16 = left.clone().cast_into();
        let right: FheInt16 = right.clone().cast_into();
        let product: FheInt32 = (&left * &right).cast_into();
        result += product;
        let completed = index + 1;
        let percent = completed * 100 / a.len();
        if show_progress && (terminal || percent >= last_percent + 5 || completed == a.len()) {
            let bar = progress_bar(completed, a.len());
            if terminal { eprint!("\rFHE: {bar}"); } else { eprintln!("FHE: {bar}"); }
            let _ = io::stderr().flush();
            last_percent = percent;
        }
    }
    if show_progress && terminal { eprintln!(); }
    Ok(result)
}

/// Return the squared norm, not its square root, in the quantized domain.
pub fn squared_norm(a: &[FheInt8]) -> Result<FheInt32, &'static str> { dot(a, a, false) }

/// Compare an encrypted dot product with a pre-scaled clear threshold.
pub fn compare(dot: &FheInt32, threshold: i32) -> FheBool { dot.ge(threshold) }

#[cfg(test)]
mod tests {
    use super::progress_bar;
    #[test]
    fn progress_uses_actual_completed_coordinates() {
        assert!(progress_bar(32, 64).contains("32/64 (50%)"));
        assert!(progress_bar(64, 64).contains("64/64 (100%)"));
    }
}
