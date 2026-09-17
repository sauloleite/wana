import bz2
import gzip
import hashlib
import json
import lzma
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wana import Example, Level, Message, Role, check_contamination
from wana.adapters.io.jsonl import JsonlReader, parse_record
from wana.adapters.matching.minhash import MinHashMatcher
from wana.adapters.matching.ngram import NgramMatcher
from wana.adapters.report.json import JsonReportWriter, serialize
from wana.application.check_contamination import check_examples
from wana.cli.main import main


def example(text, id="train"):
    return Example(id, (Message(Role.USER, text),))


def write_data(
    path, text="one two three four five six seven eight nine ten eleven twelve thirteen"
):
    path.write_text(json.dumps({"instruction": text, "output": "answer"}) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "record",
    [
        {
            "messages": [
                {"role": "user", "content": "hello\nworld"},
                {"role": "assistant", "content": "ok"},
            ]
        },
        {"instruction": "hello", "input": "world", "output": "ok"},
        {
            "conversations": [
                {"from": "human", "value": "hello\nworld"},
                {"from": "gpt", "value": "ok"},
            ]
        },
    ],
)
def test_formats(record):
    parsed = parse_record(record, "id")
    assert parsed.text == "hello\nworld\nok"
    assert parsed.messages[-1].role == Role.ASSISTANT


@pytest.mark.parametrize(
    "record",
    [
        [],
        {},
        {"messages": []},
        {"messages": "x"},
        {"messages": [1]},
        {"messages": [{"role": "unknown", "content": "x"}]},
        {"messages": [{"role": "user", "content": []}]},
        {"instruction": 1, "output": "x"},
        {"instruction": "x", "input": None, "output": "x"},
        {"instruction": "x", "output": None},
    ],
)
def test_bad_records(record):
    with pytest.raises(ValueError):
        parse_record(record, "id")


@pytest.mark.parametrize(
    "suffix,opener", [("", open), (".gz", gzip.open), (".xz", lzma.open), (".bz2", bz2.open)]
)
def test_compression_and_lines(tmp_path, suffix, opener):
    path = tmp_path / ("data.jsonl" + suffix)
    with opener(path, "wt", encoding="utf-8") as stream:
        stream.write('\n{"instruction": "olá", "output": "ação"}\n')
    rows = list(JsonlReader().read(str(path)))
    assert rows[0].id == f"{path}:2"
    assert rows[0].text == "olá\nação"


def test_error_location(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("\nnot json\n")
    with pytest.raises(ValueError, match=r"bad.jsonl:2:"):
        list(JsonlReader().read(str(path)))


def test_exact_normalization_evidence_and_order():
    matcher = NgramMatcher(2)
    hits = list(
        matcher.hits(
            [example("CAFÉ, verde azul")],
            [example("verde azul", "a"), example("cafe\u0301 verde", "b")],
        )
    )
    assert [h.eval_id for h in hits] == ["a", "b"]
    assert [h.evidence for h in hits] == ["verde azul", "café verde"]
    assert not list(NgramMatcher().hits([example("short")], [example("short", "eval")]))


@pytest.mark.parametrize(
    "matcher",
    [
        NgramMatcher(2, ignore_template=("Common template",)),
        MinHashMatcher(shingle=2, ignore_template=("Common template",)),
    ],
)
def test_ignore_template(matcher):
    assert not list(
        matcher.hits(
            [example("Common template apple")], [example("COMMON TEMPLATE orange", "eval")]
        )
    )


def test_minhash_verifies_jaccard():
    # One-row bands make candidates likely; verify against the actual shingle sets.
    matcher = MinHashMatcher(0.8, shingle=1, num_perm=64, bands=64)
    train = example("a b c d e f g h i j")
    hits = list(matcher.hits([train], [example("a b c d e f g h i k", "eval")]))
    assert len(hits) == 1
    assert hits[0].value == pytest.approx(9 / 11)
    assert hits[0].level == Level.NEAR
    strict = MinHashMatcher(0.9, shingle=1, bands=64)
    assert not list(strict.hits([train], [example("a b c d e f g h i k", "eval")]))
    assert not list(matcher.hits([example("")], [example("", "eval")]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0},
        {"threshold": 1.1},
        {"shingle": 0},
        {"num_perm": 0},
        {"bands": 0},
        {"num_perm": 7},
        {"num_perm": 65537},
    ],
)
def test_invalid_minhash(kwargs):
    with pytest.raises(ValueError):
        MinHashMatcher(**kwargs)


def test_invalid_ngram():
    with pytest.raises(ValueError):
        NgramMatcher(0)


def test_policy_injection_and_duplicate_ids():
    train, evaluation = [example("a b c")], [example("a b c", "eval")]
    assert not check_examples(train, evaluation, matchers=[NgramMatcher(2)]).ok
    assert check_examples(train, evaluation, matchers=[NgramMatcher(2)], fail_on=()).ok
    assert check_examples([], [], matchers=[]).ok
    for left, right in [(train * 2, evaluation), (train, evaluation * 2)]:
        with pytest.raises(ValueError, match="duplicate"):
            check_examples(left, right, matchers=[])

    class FakeMatcher:
        name = "fake"

        def hits(self, train, evaluation):
            return ()

    assert check_examples(train, evaluation, matchers=[FakeMatcher()]).ok


def test_public_api_and_writer(tmp_path):
    path = write_data(tmp_path / "train.jsonl")
    report = check_contamination(path, eval_sets=[str(path)])
    assert not report.ok
    assert len(report.hits) == 2
    assert check_contamination([example("a")], eval_sets=[[example("b", "eval")]]).ok
    assert check_contamination(path, eval_sets=[path], matchers=[]).ok
    output = tmp_path / "report.json"
    JsonReportWriter().write(report, str(output))
    assert output.read_text() == serialize(report)


def test_cli_manifest_and_determinism(tmp_path, capsys):
    train = write_data(tmp_path / "train.jsonl")
    evaluation = write_data(tmp_path / "valid.jsonl")
    parent = tmp_path / "manifest.json"
    parent.write_text('{"wyra_version": "0.1.1"}')
    out = tmp_path / "result"
    args = ["check", str(train), "--eval", str(evaluation), "-o", str(out)]
    assert main(args) == 1
    first = (out / "contamination.json").read_bytes()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["parent"]["sha256"] == hashlib.sha256(parent.read_bytes()).hexdigest()
    assert manifest["inputs"][0]["records"] == 1
    assert manifest["outputs"][0]["sha256"] == hashlib.sha256(first).hexdigest()
    assert main(args) == 1
    assert (out / "contamination.json").read_bytes() == first
    assert main([*args, "--fail-on"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_errors_and_empty(tmp_path, capsys):
    train = write_data(tmp_path / "train.jsonl")
    args = ["check", str(train), "--eval", str(train), "-o", str(tmp_path / "out")]
    assert main([*args, "--ngram", "0"]) == 2
    assert main([*args, "--parent", str(tmp_path / "missing")]) == 2
    parent = tmp_path / "parent.json"
    parent.write_text("[]")
    assert main([*args, "--parent", str(parent)]) == 2
    (tmp_path / "manifest.json").write_text("{}")
    assert main([*args, "-o", str(tmp_path)]) == 2
    assert "overwrite" in capsys.readouterr().err
    train.write_text("")
    assert main(args) == 0
    train.write_text("invalid")
    assert main(args) == 2


def test_cli_multiple_inputs(tmp_path):
    left = write_data(tmp_path / "a.jsonl", "left")
    right = write_data(tmp_path / "b.jsonl", "right")
    out = tmp_path / "out"
    assert main(["check", str(left), str(right), "--eval", str(left), "-o", str(out)]) == 0
    report = json.loads((out / "contamination.json").read_text())
    assert report["train_records"] == 2


@given(
    st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), min_size=1, max_size=20),
    st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), min_size=1, max_size=20),
)
@settings(max_examples=30, deadline=None)
def test_matcher_symmetry_and_determinism(left, right):
    a, b = example(" ".join(left), "a"), example(" ".join(right), "b")
    for matcher in (NgramMatcher(2), MinHashMatcher(shingle=2, num_perm=8, bands=4)):
        forward = list(matcher.hits([a], [b]))
        reverse = list(matcher.hits([b], [a]))
        assert [(h.level, h.value, h.evidence) for h in forward] == [
            (h.level, h.value, h.evidence) for h in reverse
        ]
        assert forward == list(matcher.hits([a], [b]))


def test_golden():
    report = check_contamination(
        "tests/fixtures/train.jsonl", eval_sets=["tests/fixtures/valid.jsonl"]
    )
    assert serialize(report) == Path("tests/fixtures/contamination.json").read_text()


def test_different_hash_seeds():
    code = (
        "from wana import check_contamination; "
        "from wana.adapters.report.json import serialize; "
        "print(serialize(check_contamination('tests/fixtures/train.jsonl', "
        "eval_sets=['tests/fixtures/valid.jsonl'])))"
    )
    results = [
        subprocess.check_output(
            [sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed}
        )
        for seed in ("1", "42")
    ]
    assert results[0] == results[1]


def test_module_version():
    from wana import __version__

    assert (
        subprocess.check_output([sys.executable, "-m", "wana", "--version"]).strip()
        == f"wana {__version__}".encode()
    )
