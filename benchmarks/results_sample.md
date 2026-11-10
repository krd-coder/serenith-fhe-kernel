# Initial CPU observation

The following is a single exploratory run, not a statistically representative benchmark. It uses the Rust example `rust-fhe-kernel/examples/cpu_timing.rs`; model inference and ZK proofs are not included.

| Item | Observation |
| --- | --- |
| Dimension | 64 |
| Input | Both vectors `[127, 0, …, 0]`, signed i8; no random seed |
| Threshold | 4/5, scaled to 12904 |
| Decision | `true`, matching the integer reference |
| Floating and quantized similarity | 1.0 and 1.0; absolute quantization error 0 |
| Setup | 2.759 s |
| Encryption of two vectors | 0.006 s |
| Ciphertext expansion and evaluation | 309.066 s |
| Decryption | Less than 0.0005 s at displayed precision |
| CPU | Intel Core i7-1260P, 16 logical CPUs visible |
| Environment | Linux on WSL2, 7.6 GiB memory visible, TFHE-rs 1.8.2, release build |
| Thread configuration | Default; exact worker count was not recorded |
| Repetitions / warmup | 1 / 0 |

From the repository root, the example can be rerun with:

```sh
cargo run --release --locked --example cpu_timing
```

The expanded benchmark script accepts seeds, warmups and repetitions and records individual samples and environment information. Run it separately for the desired dimensions; CPU evaluations at 256 and 768 dimensions can take substantially longer.

The current testing scope is 64 dimensions. No measurements for 256 or 768 dimensions were performed in this iteration.

## ZKV2 end-to-end check

One exploratory 64-dimensional CPU run used two identical example texts, threshold 4/5, Nomic already downloaded, TFHE-rs 1.8.2 and the optimized Rust build on Intel Core i7-1260P under WSL2. Both ciphertext proofs verified, the encrypted result decrypted to `true`, and the quantized reference decision was also `true`.

The complete command, including model loading and inference, key setup, proof generation and verification, encrypted evaluation and output, took **171.50 seconds** of wall-clock time. This is one observation rather than a benchmark distribution; it is not directly comparable to the separately timed evaluation above. The model revision and complete dependency environment were not pinned for this run.
