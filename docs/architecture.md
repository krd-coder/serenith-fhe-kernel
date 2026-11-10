# Architecture

## Purpose and scope

Serenith explores privacy-preserving similarity matching between document representations. Its broader infrastructure combines document embedding, encrypted computation and protocol components that coordinate matching and result handling.

Serenith also has a broader, separately maintained proof of concept that explores a larger part of the proposed system. This public repository presents selected ideas from that work through a focused FHE kernel and demonstration. It is not a complete release of the broader proof of concept.

The repository makes the numerical and cryptographic approach independently inspectable.

Beyond the public kernel, Serenith architecture brings together client-side document processing, homomorphic encryption, cryptographic proofs and decentralized coordination. Smart contracts are designed to coordinate selected protocol interactions, while off-chain components handle encrypted data and computation. This division allows the public kernel to focus on the cryptographic operations that underpin private similarity matching.

## System context

At a high level, the broader Serenith design consists of:

- **Client-side preparation:** a document is converted into an embedding, reduced to a supported Matryoshka Representation Learning (MRL) prefix, normalized and quantized.
- **Encrypted computation:** quantized representations are encrypted and evaluated using TFHE-based operations.
- **Protocol and coordination:** components outside this repository coordinate data availability, computation requests and result handling.
- **Verification and governance:** the broader design includes mechanisms intended to support verification and participation rules.

This public repository focuses on the preparation and encrypted computation needed for a single-key demonstration. It does not implement the full protocol, distributed key management, on-chain coordination or a production network.

The public demonstration defaults to CPU and includes an experimental CUDA path based on the broader PoC's GPU evaluation. CUDA and CPU use separate key configurations; the public CUDA path still requires validation on an NVIDIA GPU. Timings from these paths are not directly interchangeable.

```text
Broader Serenith infrastructure (simplified)
┌───────────────────────────────────────────────────────────────┐
│ Client preparation   Encrypted computation   Coordination     │
│ and encryption  ───► and matching      ───► and result use    │
└───────────────────────────────────────────────────────────────┘
         │                       │
         └──── Public demo and FHE kernel ────┘
```

## Public repository components

### Python demonstration layer

The Python demonstration follows a linear workflow:

1. Accept text inputs and produce embeddings using an MRL-compatible model.
2. Apply layer normalization to the full embedding, select a supported prefix and apply L2 normalization to that prefix.
3. Quantize the normalized values into signed 8-bit integers using scale `S = 127`.
4. Pass the quantized vectors to the Rust extension for encryption and evaluation.
5. Decrypt the matching decision and compare it with a cleartext reference decision.

The Python layer covers input validation and cleartext reference mathematics. Model loading and embedding occur in the environment running the demonstration; this repository does not by itself establish the provenance of an input document or embedding.

### Rust FHE kernel

The Rust crate exposes TFHE operations to Python through a PyO3 extension. Its responsibilities are:

- Set up the keys and cryptographic context required by the single-key demonstration.
- Encrypt validated quantized vectors.
- Optionally produce and verify a proof associated with ciphertext construction.
- Evaluate encrypted vector operations, including a widened dot product and comparison against a threshold in the quantized domain.
- Return encrypted results for decryption by the demonstration’s key holder.

The exact supported dimensions, parameter choices, numerical bounds and result representations must be documented alongside the implementation.

### Tests and benchmarks

Tests check input validation and compare decrypted FHE results against the corresponding quantized cleartext calculations. Benchmarks report reproducible measurements for the supported dimensions, including separate setup, encryption, evaluation and decryption timings.

Benchmark results describe the measured configuration; they should not be interpreted as performance guarantees for the complete Serenith infrastructure.

## Data flow

```text
Text A ─► embedding ─► layer norm ─► MRL prefix ─► L2 norm ─► quantize ─► encrypt ─┐
                                                                                    ├─► FHE evaluation
Text B ─► embedding ─► layer norm ─► MRL prefix ─► L2 norm ─► quantize ─► encrypt ─┘         │
                                                                                            ▼
                                                                             encrypted decision
                                                                                            │
                                                                                            ▼
                                                                   decrypt and compare with reference
```

For normalized vectors `a` and `b`, the cleartext similarity reference is their dot product. After quantization with scale `S`, the integer dot product is compared with a correspondingly scaled threshold. Quantization introduces approximation relative to the floating-point calculation; the FHE result is to be checked against the integer reference separately from that approximation.

## Trust boundaries and assumptions

- **Plaintext boundary:** text, embeddings and quantized vectors are available in plaintext to the environment that prepares them. FHE does not conceal inputs from that environment.
- **Key boundary:** the single-key demonstration uses a key holder capable of decrypting its result. This is a demonstration assumption, not a description of distributed key management in the broader design.
- **Evaluation boundary:** FHE evaluation operates on encrypted values using evaluation material. Its confidentiality properties depend on the selected parameters, correct key handling and the security of the underlying implementation.
- **Proof boundary:** where an optional proof of ciphertext construction is used, its precise statement and verification conditions must be documented. Such a proof must not be presented as proof that an embedding came from a particular document or that preprocessing was performed correctly unless those claims are separately established.
- **Output boundary:** decrypted scores or decisions can themselves reveal information. Who is allowed to receive them and what can be inferred from them are concerns for the surrounding protocol, beyond this demonstration.

## Status and limitations

The public kernel is in development; the architecture above describes the demonstration and its relationship to the wider system.

The demo is not a complete privacy-preserving application. In particular, encrypted computation alone does not establish document provenance, protect plaintext within the client environment, or provide the coordination and access controls of a deployed system. See `limitations.md` for implementation-specific numerical and operational limits as they are established.
