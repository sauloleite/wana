import importlib.util
from pathlib import Path

import pytest


def load_summary():
    spec = importlib.util.spec_from_file_location(
        "compare_judgments", Path("experiments/compare_judgments.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize


def test_judgment_summary_preserves_negative_and_inconclusive_results():
    summarize = load_summary()
    rows = [{"id": i, "seed": 42, "judge_model": "fixture", "winner": "random"} for i in range(20)]
    assert summarize(rows)["criterion_met"] is False
    assert summarize(rows)["non_tie_win_rate"] == 0
    for row in rows:
        row["winner"] = "tie"
    assert summarize(rows)["non_tie_win_rate"] is None
    for row in rows:
        row["winner"] = "selected"
    assert summarize(rows)["criterion_met"] is True
    with pytest.raises(ValueError):
        summarize(rows + rows)


def test_clustered_summary_handles_correlated_seeds_and_rejects_mixed_comparisons():
    spec = importlib.util.spec_from_file_location(
        "aggregate_benchmark", Path("experiments/aggregate_benchmark.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {"id": i, "seed": seed, "baseline": "random", "winner": "tie"}
        for i in range(10)
        for seed in (42, 43, 44)
    ]
    result = module.aggregate(rows, samples=100)
    assert result["prompts"] == 10
    assert result["judgments"] == 30
    assert result["prompt_cluster_bootstrap_95"] == [0.5, 0.5]
    assert result["criterion_met"] is False
    with pytest.raises(ValueError):
        module.aggregate(rows[:-1])
    with pytest.raises(ValueError):
        module.aggregate(rows + rows)
    rows[0]["baseline"] = "full"
    with pytest.raises(ValueError):
        module.aggregate(rows)


def test_paper_order_policy_keeps_unopposed_wins_and_ties_conflicts():
    spec = importlib.util.spec_from_file_location(
        "judgment_policy", Path("experiments/judgment_policy.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    combine = module.combine_orders
    for baseline in ("random", "full"):
        for first in ("selected", baseline, "tie"):
            for second in ("selected", baseline, "tie"):
                expected = (
                    first
                    if first == second
                    else second
                    if first == "tie"
                    else first
                    if second == "tie"
                    else "tie"
                )
                assert combine(first, second, baseline=baseline, policy="paper") == expected
                assert combine(first, second, baseline=baseline, policy="strict") == (
                    first if first == second else "tie"
                )
    with pytest.raises(ValueError):
        combine("selected", "full", baseline="random", policy="paper")
    with pytest.raises(ValueError):
        combine("tie", "tie", baseline="random", policy="unknown")
