# Local validation — 2026-09-15

Environment: macOS arm64, Python 3.14.7 (the CI matrix for 3.10–3.13 has not
run remotely yet).

- 39 tests passed; domain/application line coverage: 100% (minimum: 90%).
- Ruff lint/format and strict mypy passed.
- Wheel and source distribution built with Hatchling; Twine metadata checks passed.
- Wheel installed with `--no-deps` into a fresh virtual environment outside the
  checkout. `wana --version` and the public API succeeded. Metadata has no
  unconditional runtime requirements.
- Built 10 examples using local Wyra 0.1.1 and its `clean_code.md` example;
  split into 7 training and 3 validation records. Appended a validation record
  to training. Wana reported EXACT=1, NEAR=1 and exit code 1, with the Wyra
  manifest recorded as parent. Wyra writes `validation.jsonl`.
- PyPI's JSON endpoint for `wana` returned HTTP 404. This does not reserve the
  name or establish publication rights.

No package was published, no GitHub release was created and no npm name was
reserved. The local repository has no Git remote configured. Remote CI and
Trusted Publisher setup remain release prerequisites.
