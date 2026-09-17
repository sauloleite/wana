# Phase status — 2026-09-17

| Phase | Implementation / evidence | Acceptance still outstanding |
| --- | --- | --- |
| 0 | Exact PyPI 0.0.1 installation verified; 0.5.0 published on PyPI and npm as @sauloleite/wana. Twelve Python platform jobs, Node and integration passed remotely. | None for the published 0.5.0 release. |
| 1 | EXACT/NEAR, three formats, compressed input, manifest, terminal; real Wyra CLI test plants a validation record and detects it in remote CI. | None for this phase. |
| 2 | Bundled offline model, Transformers CPU/fake, IFD/auxiliary scores, top-k/stratified, cache. Alpaca-1k scoring completed; six-budget trained quality experiment in progress. | Final curve; the experiment adapts the paper and is not exact replication of its 7B setup. |
| 3 | Diverse greedy, hashing/ONNX embeddings, SEMANTIC, Markdown; two documented three-seed pilots. | Corrected selected-20% comparison: 48.3% preference, clustered 95% interval 45.5%–51.2%. Superiority over random is not established. |
| 4 | run, artifacts, parent provenance, explain/hash verification; real Wyra chain passes remotely. | None for this phase. |
| 5 | Published Node check/manifest verifier; shared golden byte equality and multilingual/near-match parity. Registry-installed CLI returns wana 0.5.0. | None for the scoped initial port; xz/bzip2 remain Python-only. |

The default pip installation intentionally includes weights, per the user's
updated request. Hashing embeddings fit corpus TF-IDF during diverse selection. Standalone CLI selection retries in 0.5.1 reuse intact same-parameter manifests;
`--reselect` and in-memory APIs apply budgets to the current input. Fixed-count
top-k is tested for idempotence. The project does not
claim all scientific or external publication acceptance criteria are complete.


## Requirements beyond the phase table

- Streaming: readers and 0.5.1 score CLI stream record bodies; scoring retains
  identities; file checks replay training and retain evaluation indexes and hits.
  Collected APIs/select/run still retain data and results.
  Fully streaming orchestration is not complete.
- Standalone fractional selection retries use verified manifests in 0.5.1.
  The in-memory API has no retry identity and still applies the current-input
  fraction. Stateless fractional selection cannot be idempotent for 0 < keep < 1.
- The optional openai_compat loss provider described in the architecture sketch
  is not implemented. Local bundled/Transformers/fake providers are implemented.
- The weekly newest-adapter workflow is added in 0.5.1; its first remote run must
  be observed before claiming that check has passed.
- npm interactive publication is verified. The optional OIDC publisher still
  needs registry configuration before the manual npm workflow can publish.
