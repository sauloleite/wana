# ADR 001: deterministic, dependency-free contamination core

Status: accepted for 0.1.0.

The initial release implements phase 1, which can be useful without models.
Domain entities are frozen dataclasses; application services depend only on
domain and Protocol ports. Concrete adapters and the public API/CLI are the
composition boundary. argparse and stdlib compression avoid runtime dependencies.
Optional scoring/embedding contracts are deferred until their implementations
can be tested. Wyra's salted BLAKE2b MinHash approach informs the matcher; actual
Jaccard verifies every candidate instead of using the MinHash estimate as evidence.

LSH trades recall for candidate-search cost. Exact n-gram indexing is exhaustive
for its defined normalization. All matching is symmetric and ordering is stable.
Small records are not silently reinterpreted as smaller n-grams.

The JSONL reader streams. This first checker materializes both datasets to replay
injected matchers and validate IDs. Fully streaming multi-matcher orchestration
is deferred and the memory limitation is public. Reports can include two levels
for a pair because they carry different evidence.

SHA-256 manifests link artifacts without claiming cryptographic authenticity.
Signing and timestamp reproducibility are separate concerns; 0.1.0 records UTC
creation time and produces deterministic report bytes. No private keys or model
credentials are needed. Future hashing embeddings will be decided in their own ADR.
