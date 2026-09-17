# Local validation — 2026-09-17

Environment: macOS arm64, Python 3.14.7. The expanded GitHub CI matrix has not
been observed remotely; local results do not establish cross-platform CI status.

- **69 Python tests passed**, including real installed SmolLM2 inference,
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
- Remote success of the expanded CI or publication of 0.5.0 to PyPI.
- First publication of @sauloleite/wana to npm (local npm client is not logged in).

Python 0.1.0 is already available on PyPI. This change prepares 0.5.0 and does
not manufacture historical 0.0.1/0.2/0.3/0.4 releases. The complete implementation
and benchmark boundaries are in [roadmap-status.md](roadmap-status.md).
