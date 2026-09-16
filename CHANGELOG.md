# Changelog

## 0.1.0 — Unreleased

- Add immutable examples, contamination evidence and provenance dataclasses.
- Read OpenAI chat, Alpaca and ShareGPT JSONL, including gzip, xz and bzip2.
- Detect shared word n-grams and MinHash/LSH candidates verified with Jaccard.
- Support literal template removal, custom matcher injection and failure policy.
- Add `wana check`, a Python API, JSON reports, terminal summary and chained manifests.
- Add golden/property tests, strict type checks and cross-platform CI.
- Publish wheel/sdist on pushes to main after CI, using Trusted Publishing and
  attestations; serialize uploads and skip existing distribution files.
