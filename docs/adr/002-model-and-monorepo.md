# ADR 002: installed scoring weights and a Python/Node monorepo

Status: accepted following the user's updated requirements (2026-09-16).

## Default model ships through pip

The user explicitly replaced the zero-dependency installation requirement with
an installation that includes the model. `wana` now depends on
`llm-smollm2==0.1.2`, whose wheel includes the complete SmolLM2-135M-Instruct Q4_1
GGUF. The model dependency wheel is 92,871,040 bytes; the GGUF SHA-256 verified
locally is `b179c9523d0e6a0f98a330c7562b682750a6f8c8c15e5bc70ea373728110db53`.
Inference uses llama-cpp-python on CPU. There is no post-install hook or
first-use model download. Native compilation may require a C/C++ toolchain on
platforms without an available runtime wheel. The existing llm-smollm2 package
also brings the LLM command-line framework transitively; Wana does not invoke it.

Selection criteria: fit within normal PyPI file limits, contain actual trained
weights, work offline, expose logits for response-only loss, and permit
redistribution. SmolLM2 and the distribution wrapper use Apache-2.0.
The MIT license on Wana does not replace third-party licenses.

| Candidate format | Observed weight/distribution size | Decision |
| --- | ---: | --- |
| SmolLM2-135M PyTorch safetensors | 269,060,552 bytes | Optional Transformers adapter |
| SmolLM2-135M ONNX int8 | 137,147,867 bytes | Larger than default PyPI file limit |
| SmolLM2-135M Q4_1 via llm-smollm2 wheel | 92,871,040 bytes | Default pip dependency |

This is a packaging and CPU-size choice, not a demonstrated optimum for IFD.
The model is mainly English-oriented. Quantization can alter rankings. The
Superfiltering paper motivates small proxy models but does not establish the
quality of this specific model/quantization. Multilingual use needs validation.
Do not use this small scorer as an authoritative downstream quality judge.

Sources:
- https://github.com/simonw/llm-smollm2
- https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- https://docs.pypi.org/project-management/storage-limits/
- https://arxiv.org/abs/2402.00530

## Scoring convention

For each example, the final assistant message is the target. Earlier messages
form its context. The model's stored chat template renders conditional context;
unconditional loss uses BOS (or EOS if no BOS exists). Both passes tokenize the
same target independently of context and exclude EOS from the loss. This avoids
inconsistent token spans across the two losses. Long examples fail explicitly;
no silent truncation. Revisions/template hashes and context limits are recorded.
The `fake` provider is solely for tests. Pure algorithms are deterministic;
floating-point scores need not be bit-identical across hardware/runtime versions.

## Two packages, shared fixtures

Python is at the repository root; Node is in `node/` and publishes independently
as `@sauloleite/wana`, as authorized by the user because npm's `wana` already exists.
Shared JSON fixtures exercise normalized n-grams, salted BLAKE2b MinHash and
Jaccard evidence. Node uses @noble/hashes for BLAKE2b with salt and digest size.
A generated full case-fold mapping handles Python/JavaScript differences; NFC
still uses the runtime's Unicode tables. Unicode-version changes require parity
validation before release. Node supports plain JSONL/gzip; Python additionally
supports xz/bzip2. IFD remains Python-only.
