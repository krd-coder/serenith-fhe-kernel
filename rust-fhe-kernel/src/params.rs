//! Numeric and cryptographic parameters shared with the Serenith PoC.
use tfhe::shortint::parameters::{
    PARAM_KEYSWITCH_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128,
    PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128,
    PARAM_PKE_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128,
};
use tfhe::{Config, ConfigBuilder};

pub const SCALE: i32 = 127;
pub const DIMENSIONS: [usize; 3] = [64, 256, 768];

/// The CPU TFHE parameters and dedicated compact-PKE parameters used in the PoC.
pub fn config() -> Config {
    ConfigBuilder::with_custom_parameters(PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128)
        .use_dedicated_compact_public_key_parameters((
            PARAM_PKE_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128,
            PARAM_KEYSWITCH_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128,
        ))
        .build()
}

/// GPU parameter profile and compact-PKE settings used by the PoC.
#[cfg(feature = "cuda")]
pub fn cuda_config() -> Config {
    use tfhe::shortint::parameters::PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128;
    ConfigBuilder::with_custom_parameters(PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128)
        .use_dedicated_compact_public_key_parameters((
            PARAM_PKE_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128,
            PARAM_KEYSWITCH_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128,
        ))
        .build()
}

pub fn validate_dimension(size: usize) -> Result<(), &'static str> {
    if DIMENSIONS.contains(&size) { Ok(()) } else { Err("dimension must be 64, 256 or 768") }
}

/// Ceiling is required for an integer dot product compared with a rational threshold.
pub fn scaled_threshold(numerator: u32, denominator: u32) -> Result<i32, &'static str> {
    if denominator == 0 || denominator > 1_000_000 || numerator > denominator {
        return Err("require 0 <= numerator <= denominator <= 1000000");
    }
    Ok((u64::from(numerator) * (SCALE as u64).pow(2)).div_ceil(u64::from(denominator)) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn threshold_and_dimensions() {
        assert_eq!(scaled_threshold(4, 5), Ok(12904));
        assert!(scaled_threshold(1, 0).is_err());
        assert!(validate_dimension(128).is_err());
        for d in DIMENSIONS { assert!(validate_dimension(d).is_ok()); }
    }
}
