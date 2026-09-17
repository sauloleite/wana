# Local validation — 2026-09-17

Local environment: macOS arm64, Python 3.14.7. The [0.5.0 CI run](https://github.com/sauloleite/wana/actions/runs/35279434345)
passed all twelve Python 3.10–3.13 Linux/macOS/Windows jobs, Node, real adapter
integration, build and PyPI publication.

- **72 Python tests passed**, including real installed SmolLM2 inference,
  Transformers CPU on a pinned local snapshot, ONNX embeddings and Wyra CLI
  integration. No skipped tests when both local model paths are supplied.
- Domain/application coverage: **100%**, minimum required 90%.
- Ruff lint/format and strict mypy passed.
- **4 Node tests passed**: golden report byte equality, normalization/validation,
  six shared multilingual/near-match cases, and manifest tamper detection.
- The Python wheel/source distribution and scoped npm tarball build successfully;
  Twine validates Python metadata. Weights/data/cache are not accidentally bundled
  into Wana's own artifacts; the default model arrives through its declared dependency.
- A clean virtual environment installed the built wheel with dependencies; the
  default model ran offline through `wana run`, `explain` verified all artifacts,
  and the Node verifier accepted the same Python manifest.
- Real Wyra 0.1.1 CLI produces a train/validation split; the test appends a validation
  record to training, runs the pipeline and confirms evidence plus the parent hash.
- Actual Alpaca-1k run: 1,000 scored, zero errors, median IFD 0.68337, 2 above 1,
  178.23 seconds. 200 selected and 200 random examples plus 100 disjoint evaluation
  records were prepared. EXACT/NEAR found no contamination in that experimental split.

- Trained-model pilot completed: nine fine-tunes (three arms × three seeds), 900
  generated answers, and 600 counterbalanced paired judgments. Selected versus
  random: 16 wins / 20 losses / 264 ties; preference 49.3%, prompt-cluster 95%
  interval 46.8%–51.8%. Selected versus full: 43 / 24 / 233; 53.2%, interval
  49.5%–56.8%. Neither comparison establishes superiority. See the
  [full report](../experiments/results/benchmark/README.md).

## Not established

- Reproduction of the paper's downstream trained-model quality curve.
- A judged trained-model result demonstrating that selected 20% beats random.

Python 0.0.1 was published as an explicit compatibility bootstrap from the
original skeleton, after 0.1.0. A fresh environment installed it from PyPI and
returned `wana 0.0.1`. Python 0.5.0 and npm @sauloleite/wana 0.5.0 are published.
The npm registry install returned `wana 0.5.0`. No historical 0.2/0.3/0.4
releases are manufactured. The complete implementation
and benchmark boundaries are in [roadmap-status.md](roadmap-status.md).

## Corrected pilot

The fixed second protocol adds EOS targets and the paper's two-order judge
aggregation, with 100 new evaluation prompts excluded from both training and
the old evaluation. Three seeds yielded 29 wins, 39 losses and 232 ties against
random; preference 48.3%, clustered 95% interval 45.5%–51.2%. The acceptance
criterion remains unmet. See [primary summary](../experiments/results/retention-curve/primary.summary.json).

A 20-record training-only consistency audit compared bundled quantized IFD
against Transformers on the same model: Spearman 0.97744. This is evidence of
ranking consistency on those records, not downstream superiority.

## 0.5.1 scoring change

The CLI scores and writes incrementally. Tests verify lazy consumption and
that a late scoring failure preserves the previous output and removes the
temporary file. The collecting API is retained. Checks from files replay training one record at a time per matcher. Evaluation,
IDs and hits remain in memory, as do collecting APIs, selection and run.
The CLI selection retry test verifies unchanged bytes across three output paths,
explicit reselection, changed budgets and rejection of a modified previous artifact.
