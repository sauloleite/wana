# Phase status — 2026-09-17

| Phase | Implementation / evidence | Acceptance still outstanding |
| --- | --- | --- |
| 0 | Python 0.1.0 already on PyPI; package/CLI/CI exist. User authorized npm name @sauloleite/wana because wana is occupied. | npm first publication requires account access. 0.0.1 compatibility wheel and manual publishing workflow are prepared and locally tested; registry publication is pending. |
| 1 | EXACT/NEAR, three formats, compressed input, manifest, terminal; real Wyra CLI test plants a validation record and detects it. | Remote CI must execute the expanded workflow. |
| 2 | Bundled offline model, Transformers CPU/fake, IFD/auxiliary scores, top-k/stratified, cache. Actual Alpaca-1k scoring completed. | The local three-arm pilot does not reproduce the paper’s downstream quality curve over multiple retention budgets. |
| 3 | Diverse greedy, hashing/ONNX embeddings, opt-in SEMANTIC, Markdown. Three-seed trained-model pilot completed: 300 paired judgments against random and 300 against full. | The selected 20% did not demonstrate superiority over random: 49.3% preference, clustered 95% interval 46.8%–51.8%. This acceptance criterion is not met. |
| 4 | run, score/select artifacts, parent provenance, explain/hash verification. Real Wyra -> run integration passes locally. | Expanded release workflow not yet observed remotely. |
| 5 | Node check/manifest verifier, shared golden byte equality and multilingual/near-match JSON parity. | npm publication and broader runtime matrix. Node xz/bzip2 input is outside this initial port. |

The default pip installation intentionally includes weights, per the user's
updated request. Hashing embeddings fit corpus TF-IDF during diverse selection. Fractional selection applies to the current input and is
not idempotent; fixed-count top-k is tested for idempotence. The project does not
claim all scientific or external publication acceptance criteria are complete.
