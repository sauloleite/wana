import json
import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wana import Example, Message, Role, run, score_dataset, select_subset, verify_manifest
from wana.adapters.embedding.fake import FakeEmbedder
from wana.adapters.embedding.hashing import HashingEmbedder
from wana.adapters.io.jsonl import parse_record
from wana.adapters.io.scored import write_scored
from wana.adapters.logprob.cache import CachedLogProbProvider
from wana.adapters.logprob.fake import FakeLogProbProvider
from wana.adapters.matching.embedding import EmbeddingMatcher
from wana.adapters.report.markdown import render_markdown
from wana.adapters.scoring.ifd import IFDScorer
from wana.adapters.scoring.length import LengthScorer
from wana.adapters.selection.diverse_greedy import DiverseGreedySelector
from wana.adapters.selection.stratified import StratifiedSelector
from wana.adapters.selection.topk import TopKSelector
from wana.application.check_contamination import check_examples
from wana.application.score_dataset import score_examples
from wana.cli.main import main
from wana.domain.score import Score, ScoredExample
from wana.domain.selection import Budget
from wana.domain.vectors import cosine
from wana.registry import Registry


def example(index=0, answer="the answer", source="a"):
    return parse_record(
        {"instruction": f"question {index}", "output": answer, "source": source}, str(index)
    )


def row(index, value, source="a"):
    return ScoredExample(example(index, source=source), (Score("ifd", value), Score("turns", 1)))


def dataset(path):
    path.write_text("\n".join(json.dumps(example(i).original) for i in range(10)) + "\n")
    return path


def test_ifd_and_auxiliary():
    result = score_dataset(
        [example()], scorers=[LengthScorer(), IFDScorer(FakeLogProbProvider(1, 2))]
    )
    assert result.examples[0].value("ifd") == 0.5
    assert result.examples[0].value("length") == 2
    assert result.examples[0].value("turns") == 1
    assert IFDScorer(FakeLogProbProvider(3, 2)).score(example())[0].flags == ("ifd_above_1",)
    with pytest.raises(ValueError, match="missing score"):
        result.examples[0].value("unknown")


@pytest.mark.parametrize(
    "conditional,unconditional", [(-1, 2), (1, 0), (math.inf, 2), (1, math.nan)]
)
def test_invalid_losses(conditional, unconditional):
    with pytest.raises(ValueError):
        IFDScorer(FakeLogProbProvider(conditional, unconditional)).score(example())


def test_invalid_scoring():
    scorer = IFDScorer(FakeLogProbProvider())
    for e in [example(answer=""), Example("x", (Message(Role.USER, "x"),)), Example("x", ())]:
        with pytest.raises(ValueError):
            scorer.score(e)
    with pytest.raises(ValueError, match="duplicate example"):
        score_examples([example(), example()], [scorer])
    with pytest.raises(ValueError, match="duplicate score"):
        score_examples([example()], [scorer, scorer])
    with pytest.raises(ValueError):
        Score("bad", math.inf)


@pytest.mark.parametrize("keep", [-1, -0.1, 1.1, math.nan, True])
def test_invalid_budgets(keep):
    with pytest.raises(ValueError):
        Budget(keep)


def test_topk_order_ties_exclusions():
    rows = [row(0, 0.5), row(1, 1.1), row(2, 0.9), row(3, 0.9)]
    selected = select_subset(rows, keep=2)
    assert [r.example.id for r in selected.examples] == ["2", "3"]
    assert selected.decisions[1].reason == "ifd_above_1"
    assert len(select_subset(rows, keep=0).examples) == 0
    assert len(select_subset(rows, keep=1.0).examples) == 3
    assert (
        len(select_subset(rows, keep=1, selector=TopKSelector(exclude_above_one=False)).examples)
        == 1
    )
    with pytest.raises(ValueError, match="duplicate"):
        select_subset([rows[0], rows[0]], keep=1)


@given(
    st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), max_size=20), st.integers(0, 20)
)
def test_topk_absolute_budget_idempotence(values, keep):
    rows = [row(i, v) for i, v in enumerate(values)]
    first = select_subset(rows, keep=keep)
    assert select_subset(first.examples, keep=keep).examples == first.examples
    assert first == select_subset(rows, keep=keep)


def test_stratified_allocation():
    rows = tuple(row(i, 0.5, "b" if i >= 7 else "a") for i in range(10))
    selected = select_subset(rows, keep=0.5, selector=StratifiedSelector())
    assert [r.example.metadata["source"] for r in selected.examples] == ["a"] * 4 + ["b"]
    assert (
        len(select_subset(rows, keep=3, selector=StratifiedSelector(strata="turns")).examples) == 3
    )
    assert not select_subset([], keep=2, selector=StratifiedSelector()).examples
    with pytest.raises(ValueError, match="missing stratum"):
        select_subset(rows, keep=3, selector=StratifiedSelector(strata="absent"))


def test_vectors_diversity_and_semantic():
    a, b, c = row(0, 0.9), row(1, 0.8), row(2, 0.7)
    embedder = FakeEmbedder(
        {a.example.text: (1.0, 0.0), b.example.text: (1.0, 0.0), c.example.text: (0.0, 1.0)}
    )
    selected = select_subset([a, b, c], keep=3, selector=DiverseGreedySelector(embedder))
    assert [r.example.id for r in selected.examples] == ["0", "2"]
    assert selected.decisions[1].reason == "similarity_threshold"
    report = check_examples([a.example], [b.example], matchers=[EmbeddingMatcher(embedder)])
    assert len(report.hits) == 1
    assert report.ok  # default failure policy is EXACT/NEAR
    assert "SEMANTIC" in render_markdown(report)
    assert cosine((0.0, 0.0), (1.0, 0.0)) == 0
    for left, right in [((), ()), ((1.0,), (1.0, 0.0)), ((math.nan,), (1.0,))]:
        with pytest.raises(ValueError):
            cosine(left, right)
    with pytest.raises(ValueError):
        DiverseGreedySelector(embedder, threshold=2)
    with pytest.raises(ValueError):
        EmbeddingMatcher(embedder, threshold=-2)
    with pytest.raises(ValueError):
        HashingEmbedder(0)
    hashing = HashingEmbedder()
    assert hashing.embed("CAFÉ café") == hashing.embed("cafe\u0301 café")
    assert sum(hashing.embed("")) == 0


def test_cache_identity(tmp_path):
    prefix = example().messages[:-1]
    cache = CachedLogProbProvider(FakeLogProbProvider(), tmp_path / "cache.sqlite", "first")
    assert cache.mean_nll(prefix, "x") == 1
    cache.provider = FakeLogProbProvider(9, 9)
    assert cache.mean_nll(prefix, "x") == 1
    cache.close()
    other = CachedLogProbProvider(FakeLogProbProvider(3, 4), tmp_path / "cache.sqlite", "other")
    assert other.mean_nll(prefix, "x") == 3
    other.close()


def test_roundtrip_preserves_original(tmp_path):
    original = {"instruction": "hi", "output": "ok", "source": "x", "extra": {"a": [1, 2]}}
    scored = score_dataset([parse_record(original, "id")], scorers=[LengthScorer()])
    output = tmp_path / "scored.jsonl"
    write_scored(output, scored.examples)
    decoded = json.loads(output.read_text())
    assert {k: v for k, v in decoded.items() if k != "wana"} == original
    assert select_subset(output, keep=1, by="length").examples[0].value("length") == 1


def test_pipeline_and_manifest_tamper(tmp_path):
    source = dataset(tmp_path / "train.jsonl")
    (tmp_path / "manifest.json").write_text('{"wyra_version":"0.1.1"}')
    result = run(
        source,
        out_dir=tmp_path / "out",
        keep=0.2,
        scorers=[LengthScorer()],
        selector=TopKSelector("length"),
    )
    assert len(result.pipeline.selection.examples) == 2
    assert result.manifest.parent is not None
    path = tmp_path / "out/manifest.json"
    assert verify_manifest(path).ok
    (tmp_path / "out/selected.jsonl").write_text("tampered")
    assert not verify_manifest(path).ok
    source.unlink()
    assert any("unreadable" in error for error in verify_manifest(path).errors)
    assert (
        run([example()], keep=1, scorers=[LengthScorer()], selector=TopKSelector("length")).manifest
        is None
    )


def test_commands(tmp_path, capsys):
    source = dataset(tmp_path / "train.jsonl")
    scored = tmp_path / "scored.jsonl"
    assert (
        main(
            [
                "score",
                str(source),
                "-o",
                str(scored),
                "--provider",
                "fake",
                "--resume",
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "select",
                str(scored),
                "-o",
                str(tmp_path / "selected"),
                "--keep",
                ".2",
                "--selector",
                "stratified",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "run",
                str(source),
                "-o",
                str(tmp_path / "run"),
                "--provider",
                "fake",
                "--keep",
                "2",
                "--diverse",
            ]
        )
        == 0
    )
    assert main(["explain", str(tmp_path / "run/manifest.json")]) == 0
    assert main(["explain", str(tmp_path / "run/manifest.json"), "--json"]) == 0
    assert main(["score", str(source), "-o", str(source), "--provider", "fake"]) == 2
    assert main(["select", str(scored), "-o", str(tmp_path / "other"), "--keep", "-1"]) == 2
    assert "Verification: PASS" in capsys.readouterr().out


def test_registry():
    registry = Registry()
    registry.register("length", LengthScorer)
    assert isinstance(registry.create("length"), LengthScorer)
    with pytest.raises(ValueError):
        registry.register("length", LengthScorer)
    with pytest.raises(ValueError):
        registry.create("other")


def test_shared_parity_fixtures():
    from wana.adapters.matching.minhash import MinHashMatcher
    from wana.adapters.matching.ngram import NgramMatcher
    from wana.adapters.report.json import serialize

    for case in json.loads(Path("tests/fixtures/parity.json").read_text(encoding="utf-8")):
        options = case["options"]
        report = check_examples(
            [parse_record(case["train"][0], "a")],
            [parse_record(case["evaluation"][0], "b")],
            matchers=[
                NgramMatcher(options["ngram"], ignore_template=tuple(options["ignoreTemplate"])),
                MinHashMatcher(
                    options["threshold"],
                    shingle=options["shingle"],
                    num_perm=options["numPerm"],
                    bands=options["bands"],
                    ignore_template=tuple(options["ignoreTemplate"]),
                ),
            ],
        )
        assert json.loads(serialize(report)) == case["report"]


def test_saved_flags_and_invalid_scores(tmp_path):
    from wana.adapters.io.scored import from_example

    scored = ScoredExample(example(), (Score("ifd", 1.1, ("ifd_above_1",)),))
    path = tmp_path / "scored.jsonl"
    write_scored(path, [scored])
    selected = select_subset(path, keep=1, selector=TopKSelector(exclude_above_one=False))
    assert selected.examples[0].scores[0].flags == ("ifd_above_1",)
    for block in [[], {"scores": []}, {"scores": {"ifd": None}}, {"flags": "bad"}]:
        e = parse_record({**example().original, "wana": block}, "bad")
        with pytest.raises(ValueError):
            from_example(e)


def test_semantic_failure_policy_requires_detector(tmp_path, capsys):
    source = dataset(tmp_path / "train.jsonl")
    assert main(["check", str(source), "--eval", str(source), "--fail-on", "SEMANTIC"]) == 2
    assert "requires --semantic" in capsys.readouterr().err


def test_streaming_scores_are_lazy_and_match_collected_results():
    from wana import iter_scored

    consumed = []

    def source():
        for i in range(3):
            consumed.append(i)
            yield example(i)

    stream = iter_scored(source(), scorers=[LengthScorer()])
    assert consumed == []
    first = next(stream)
    assert consumed == [0]
    assert (first, *stream) == score_dataset(
        [example(i) for i in range(3)], scorers=[LengthScorer()]
    ).examples


def test_scored_write_preserves_existing_file_on_late_failure(tmp_path):
    target = tmp_path / "scored.jsonl"
    target.write_text("previous result\n")

    def broken():
        yield row(0, 0.5)
        raise ValueError("late scoring failure")

    with pytest.raises(ValueError, match="late scoring failure"):
        write_scored(target, broken())
    assert target.read_text() == "previous result\n"
    assert list(tmp_path.iterdir()) == [target]
    assert write_scored(target, [row(0, 0.5)]) == 1
    assert json.loads(target.read_text())["wana"]["scores"]["ifd"] == 0.5


def test_fractional_cli_retry_reuses_verified_bytes_and_allows_reselection(tmp_path):
    source = dataset(tmp_path / "train.jsonl")
    scored = tmp_path / "scored.jsonl"
    assert main(["score", str(source), "-o", str(scored), "--provider", "fake"]) == 0
    selected = tmp_path / "first.jsonl"
    assert main(["select", str(scored), "-o", str(selected), "--keep", ".2"]) == 0
    assert len(selected.read_text().splitlines()) == 2
    repeated = tmp_path / "second.jsonl"
    assert main(["select", str(selected), "-o", str(repeated), "--keep", ".2"]) == 0
    assert repeated.read_bytes() == selected.read_bytes()
    manifest = tmp_path / "second.manifest.json"
    assert verify_manifest(manifest).ok
    assert json.loads(manifest.read_text())["steps"][0]["reused"] is True
    third = tmp_path / "third.jsonl"
    assert main(["select", str(repeated), "-o", str(third), "--keep", ".2"]) == 0
    assert third.read_bytes() == selected.read_bytes()
    smaller = tmp_path / "smaller.jsonl"
    assert main(["select", str(selected), "-o", str(smaller), "--keep", ".2", "--reselect"]) == 0
    assert smaller.read_bytes() == b""
    changed = tmp_path / "changed.jsonl"
    assert main(["select", str(selected), "-o", str(changed), "--keep", ".5"]) == 0
    assert len(changed.read_text().splitlines()) == 1
    selected.write_text(selected.read_text() + "\n")
    assert main(["select", str(selected), "-o", str(repeated), "--keep", ".2"]) == 2
    assert repeated.read_bytes() == third.read_bytes()
