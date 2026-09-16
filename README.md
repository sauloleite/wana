# wana

Deterministic contamination audits and provenance for fine-tuning datasets.
**Python 3.10+ · zero runtime dependencies · MIT**

“Wana” means “Path” in WANYAM, an extinct indigenous language of the Txapacura
family. The project connects fine-tuning datasets to evidence for their curation.

## Status

This repository implements **0.1.0, phase 1** of the [construction plan](docs/construction-plan.md):
EXACT/NEAR contamination checks, three input formats, compressed JSONL, JSON
reports, terminal summaries and chained manifests. PyPI publication requires
completing the [release setup](docs/publishing.md).

Scoring (IFD), subset selection, embeddings, `run`, `explain`, caching and the
Node package are future phases. No model extras are advertised before their
adapters exist. No training-quality or benchmark result is claimed.

## Install from source

```sh
python -m pip install .
wana --version
```

After publication, install the release with `pip install wana==0.1.0`.

## Check contamination

```sh
wana check dataset/train.jsonl --eval dataset/valid.jsonl -o audit/
wana check train-a.jsonl train-b.jsonl --eval valid.jsonl --eval golden.jsonl -o audit/
wana check train.jsonl.gz --eval valid.jsonl.xz --ignore-template 'Answer the following question:'
```

The command writes `audit/contamination.json` and `audit/manifest.json` even
when overlap causes a failing exit code. Existing reports in the output directory
are replaced; source files and the detected parent manifest are protected.

- **0:** no hit violates the selected policy.
- **1:** a hit violates the policy (EXACT and NEAR by default).
- **2:** invalid arguments, malformed data or filesystem error.

`--fail-on EXACT` only fails on EXACT; an empty `--fail-on` records evidence
without failing. Use `--ngram 13`, `--shingle 5` and `--threshold 0.8` to configure
matching. Repeated `--ignore-template TEXT` removes literal text, after Unicode
NFC normalization and case folding. It is not a regular expression or a template
parser. Review evidence before excluding examples.

The parent is auto-detected as `manifest.json` beside the first training input,
or supplied with `--parent PATH`. It may be a Wyra manifest. Every input,
evaluation set, parent and output report receives a SHA-256 digest. Parameters,
record counts and the package version are recorded automatically.

With Wyra 0.1.1, the validation filename is `validation.jsonl`:

```sh
wyra build docs/*.md -o dataset --valid 0.1
wana check dataset/train.jsonl --eval dataset/validation.jsonl -o audit/
```

```python
from wana import check_contamination

report = check_contamination("train.jsonl", eval_sets=["valid.jsonl"])
for hit in report.hits:
    print(hit.train_id, hit.eval_id, hit.level.value, hit.value, hit.evidence)
assert report.ok
```

The API also accepts iterables of immutable `Example` objects. It returns a
`Report` dataclass and does not write files. Inject custom implementations of
`Matcher` using `matchers=[...]`; inheritance is unnecessary. Loaded examples
must have unique IDs within each side of the comparison.

## Input formats

One object per nonblank line, automatically detected per record:

```jsonl
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
{"instruction":"Question","input":"Optional context","output":"Answer"}
{"conversations":[{"from":"human","value":"Question"},{"from":"gpt","value":"Answer"}]}
```

Plain UTF-8 JSONL and `.gz`, `.xz`, `.bz2` are supported. IDs use the supplied
path and physical line number, so an evidence pair locates the source records.
Text-only message content is required; multimodal payloads, null content and
unknown roles raise contextual errors rather than silently discarding content.
Extra top-level metadata is not used for matching. All message contents,
including system messages, are compared; formatting and role labels are excluded.

## Detection and limits

- **EXACT:** at least one shared normalized word 13-gram, using an inverted
  index. The score `1.0` means a shared n-gram, not identical whole records.
- **NEAR:** deterministic BLAKE2b MinHash with 64 permutations and 16 LSH bands
  proposes pairs; actual shingle Jaccard must reach the threshold. LSH is
  approximate and can miss pairs. Both EXACT and NEAR may describe the same pair.
- **SEMANTIC:** not checked in 0.1.0. Paraphrases and translations may escape.

Records shorter than each matcher's n-gram/shingle size produce no hit for that
matcher. Empty datasets are valid and report zero records. `report.ok` means
no configured detector found a policy violation; it does not prove absence of
leakage. Evidence contains source text and should be handled like the datasets.

The reader streams, but the checker currently materializes datasets, evaluation
indexes and hits in memory. Large corpora need memory proportional to input
and matches. No network or model download occurs.

Reports are deterministic for the same input bytes, path strings, ordering and
parameters, independent of Python's hash seed. JSON files use UTF-8 and LF across
platforms. Manifest timestamps vary; output report digests remain stable.
Moving inputs changes path-based IDs. The manifest hashes raw compressed bytes
and does not sign or authenticate content.

## Development

```sh
python -m venv .venv
# Activate your virtual environment, then:
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
mypy wana
pytest --cov --cov-report=term-missing
python -m build
python -m twine check dist/*
```

CI targets Python 3.10–3.13 on Linux, macOS and Windows. Domain/application
coverage must remain at least 90%. Golden fixtures, property tests and subprocess
checks exercise formats, evidence, determinism and matcher symmetry. See
[architecture decisions](docs/adr/001-core.md), [releases](docs/publishing.md)
and [changelog](CHANGELOG.md).
