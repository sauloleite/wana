# Changelog

## 0.5.0 — Unreleased

- Install SmolLM2-135M-Instruct Q4_1 weights through the default pip dependency,
  following the updated installation requirement; default scoring works offline.
- Add response-only IFD with bundled, Transformers CPU and fake providers, and
  a content/model-addressed SQLite resume cache.
- Add auxiliary scores, top-k, stratified and TF-IDF/ONNX diverse selection,
  preserving original records and KEEP/DROP reasons.
- Add opt-in semantic matching, Markdown reports, pipeline artifacts, parent
  manifests, explain and hash verification.
- Add @sauloleite/wana Node package, shared Python/Node golden tests and separate
  npm publication workflow.
- Record Alpaca-1k scoring and a completed three-seed, nine-training-run pilot.
  The judged experiment did not establish superiority over random selection;
  publish aggregate results, individual judgment labels and reproducible scripts.

## 0.1.0

- Add immutable examples, contamination evidence and provenance dataclasses.
- Read OpenAI chat, Alpaca and ShareGPT JSONL, including gzip, xz and bzip2.
- Detect shared word n-grams and MinHash/LSH candidates verified with Jaccard.
- Support literal template removal, custom matcher injection and failure policy.
- Add `wana check`, a Python API, JSON reports, terminal summary and chained manifests.
- Add golden/property tests, strict type checks and cross-platform CI.
- Publish wheel/sdist on pushes to main after CI, using Trusted Publishing and
  attestations; serialize uploads and skip existing distribution files.
