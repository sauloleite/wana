# wana

Audit, score and select fine-tuning datasets with evidence and file provenance.
Python 3.10+ · Python and Node packages in one repository · MIT library.

“Wana” means “Path” in WANYAM, an extinct indigenous language of the Txapacura
family. Wana connects training datasets to evidence for their curation.

## Release status

**Python 0.5.0 and @sauloleite/wana 0.5.0 are published.** This checkout prepares
Python 0.5.1. The Python 3.10–3.13 matrix passed on Linux/macOS/Windows, including
real adapter and Wyra integration; the npm package was installed from the registry.
See [validation](docs/validation.md) and [phase status](docs/roadmap-status.md).

The corrected three-seed pilot did **not establish superiority over random**
(48.3% preference, clustered 95% interval 45.5%–51.2%). The broader retention
curve is being completed; publication is not proof of scientific acceptance.

## Install

For this checkout:

```sh
python -m pip install .
wana --version
```

Install the published release with `pip install wana==0.5.0`.

**Normal installation includes the scoring model weights.** Wana depends on
`llm-smollm2==0.1.2`, which bundles SmolLM2-135M-Instruct Q4_1 in a roughly 93 MB
wheel, and llama-cpp-python. Once installed, default scoring is offline. Native
runtime installation may need C/C++ build tools. There are no first-use model
downloads or post-install scripts in Wana.

The model is small and predominantly English-oriented. Quantization and model
choice affect IFD ranking; IFD is not an accuracy or correctness score. The
[model decision](docs/adr/002-model-and-monorepo.md) compares packaging options,
limitations and licenses. This user-requested default supersedes the original
plan's zero-dependency installation.

Optional adapters:

```sh
pip install 'wana[score]'   # Transformers + PyTorch; choose another scoring model
pip install 'wana[embed]'   # ONNX Runtime + tokenizer for local sentence embeddings
pip install 'wana[tokens]'  # tiktoken counter for the Python API
pip install 'wana[all]'
```

## CLI

```sh
# Audit overlap. Exit 1 on EXACT or NEAR hits, 2 on invalid input/I/O.
wana check train.jsonl --eval valid.jsonl -o audit/ --markdown

# Default: installed SmolLM2 model. --resume caches losses by content and model identity.
wana score train.jsonl -o scored.jsonl --resume
wana score train.jsonl -o scored.jsonl --provider transformers_cpu \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --revision MODEL_COMMIT

# --by is the ranking score; --strata is the metadata field.
wana select scored.jsonl -o selected/ --keep 0.2 --by ifd
wana select scored.jsonl -o selected/ --keep 200 --selector stratified --strata source
wana select scored.jsonl -o selected/ --keep 0.2 --diverse --threshold 0.9

# Full pipeline, with parent manifest automatically linked.
wyra build docs/*.md -o dataset --valid 0.1
wana run dataset/ -o selected/ --eval dataset/validation.jsonl --keep 0.2 --resume
wana explain selected/manifest.json
```

`--provider fake` is for tests only. `--scorer length` skips model execution;
use `--by length` when selecting those scores. `WANA_PROVIDER`, `WANA_MODEL`
and `WANA_CACHE_DIR` supply defaults. CPU context defaults to 2048 tokens; long
examples fail explicitly, and `--max-tokens` configures the limit. IFD uses the
final assistant response and the preceding messages as context. Scores above
1 are flagged and excluded by default; `--include-ifd-above-one` retains them.

`--keep 1` means one record; `--keep 1.0` means all. Fractional budgets round
down. In 0.5.1, repeating a standalone `select` with the same parameters and
version reuses its verified sidecar manifest and copies the selected bytes.
Use `--reselect` to apply the budget to the reduced input again. Missing or
different provenance uses the ordinary current-input budget. The in-memory API
also applies its budget to the current input; fixed-count top-k is idempotent. Ties use original order, independent of seed.
Stratified selection uses largest-remainder quotas over eligible records.
Diversity accepts only examples below the cosine similarity threshold; it may
return fewer than the requested budget. Default hashing embeddings are lexical
TF-IDF vectors fitted on the input corpus, not semantic representations.

### Semantic check (opt-in)

```sh
wana check train.jsonl --eval valid.jsonl -o audit/ --semantic \
  --embedding-model /path/to/model_quantized.onnx --tokenizer /path/to/tokenizer.json \
  --semantic-threshold 0.9 --fail-on EXACT NEAR SEMANTIC
```

Supply a local ONNX sentence encoder with `input_ids` and optional
`attention_mask`/`token_type_ids`, and a Hugging Face tokenizer JSON. The adapter
supports pooled sentence output or attention-mask mean pooling of token output.
It was tested with a pinned all-MiniLM-L6-v2 ONNX snapshot. Embedding tokenization
truncates to 512 tokens. `select` also accepts `--embedder onnx` with those paths.

## Artifacts and Python API

`run` writes:

| File | Content |
| --- | --- |
| `scored.jsonl` | Original records + `wana.scores`, flags and every KEEP/DROP reason |
| `selected.jsonl` | Kept records, in original order, with annotations |
| `contamination.json` | Evidence against evaluation sets for the selected subset |
| `manifest.json` | Input/output SHA-256, parameters and optional parent |
| `report.md` | Human-readable contamination evidence |

Standalone `score` and `select` write `<output-stem>.manifest.json`; `select`
also writes `<output-stem>.decisions.json` for all input records. A sidecar
manifest beside the input takes precedence over its directory's `manifest.json`.
A failing `run` writes artifacts for inspection; it does not automatically delete
contaminated records. Outputs in the destination are replaced; input and parent
paths are protected against collisions.

```python
from wana import check_contamination, score_dataset, select_subset, run, verify_manifest

scored = score_dataset("train.jsonl")
selection = select_subset(scored, keep=0.2)
report = check_contamination([row.example for row in selection.examples], eval_sets=["valid.jsonl"])
result = run("dataset/", out_dir="selected", eval_sets=["dataset/validation.jsonl"])
assert verify_manifest("selected/manifest.json").ok
```

To score incrementally in Python:

```python
from wana import iter_scored
from wana.adapters.io.scored import write_scored
from pathlib import Path

write_scored(Path("scored.jsonl"), iter_scored("train.jsonl"))
```

Public operations accept paths or loaded dataclasses. Scorers, selectors and
matchers are injected through small Protocol interfaces. Custom components can
use `Registry` factories without inheritance. An in-memory `run` returns typed
results; it does not invent file input hashes or a file manifest.

Manifest verification checks direct artifact hashes and the parent file, not
cryptographic authenticity or recursive upstream integrity. New pipeline
manifests use absolute file paths and need those files to remain accessible.
The 0.1 check format retains cwd-relative paths. Creation timestamps vary.
Original dataset metadata is retained; reports contain source text.

## Formats, determinism and limits

Automatically detects text-only OpenAI chat (`messages`), Alpaca
(`instruction`, optional `input`, `output`) and ShareGPT (`conversations`).
Python reads UTF-8 JSONL, gzip, xz and bzip2. Invalid content and unknown roles
raise contextual errors. All message contents are compared, including system
messages. `--ignore-template TEXT` removes repeated literal text after NFC/case
normalization; it is not a regex or a template parser.

EXACT means a shared normalized word n-gram (default 13), not identical whole
records. NEAR uses 64 MinHash permutations/16 LSH bands and verifies candidates
with exact shingle Jaccard (default 5-word shingles, threshold 0.8). LSH can miss
near matches. Records shorter than each matcher size produce no hits at that
level. A pair may carry evidence from multiple levels. Empty datasets are valid.
`report.ok` means no detector found a configured violation, not proven absence
of leakage. The default failure policy excludes SEMANTIC unless requested.

Readers and the `score` CLI process records incrementally. Python callers can
use `iter_scored(source)` for the same behavior; `score_dataset` collects results
for convenience. Scoring retains IDs to detect duplicates. File-based checks replay training for each matcher without retaining training
record bodies; evaluation indexes and evidence remain in memory. Selection
and `run` still materialize datasets and results; memory grows with inputs
and matches. Scored JSONL replacement is atomic if scoring or writing fails. Pure reports are
stable for the same path strings, order, parameters and supported Unicode
normalization; model floating-point losses can vary across platforms.

## Node package

`node/` contains **@sauloleite/wana**. It implements EXACT/NEAR checks and manifest
verification, with shared Python/Node fixtures. It supports plain JSONL/gzip and
does not include IFD, embeddings or a model. See [Node README](node/README.md).
Python and npm publication are independent. Install with
`npm install @sauloleite/wana` or run `npx @sauloleite/wana --version`.

## Development

```sh
pip install -e '.[dev]'
ruff check .
ruff format --check .
mypy wana
pytest --cov --cov-report=term-missing
python -m build
python -m twine check dist/wana-0.5.1*
cd node && npm install && npm test
```

CI covers Python 3.10–3.13 on Linux/macOS/Windows, shared Node goldens, real
Wyra integration and an optional-runtime Linux job. PyPI publication runs on
pushes to `main` after those jobs pass. [Publication setup](docs/publishing.md).
