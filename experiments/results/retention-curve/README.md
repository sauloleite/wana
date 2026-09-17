# Corrected retention-budget pilot

Status: running. The fixed [plan](plan.json) was written before judging the new
outputs. It preserves the previous experiment and excludes its evaluation records
from this study's held-out set.

Corrections: train the end-of-response token; measure multiple retention budgets;
combine two answer orders using the paper's rule (win + tie is a win, opposing
wins are a tie). The primary comparison is selected 20% versus random 20% across
three seeds. The budget curve is a separate exploratory comparison to the full
arm with seed 42, at 10%, 20%, 40%, 60%, 80% and 100% retention.

Run: `python experiments/retention_curve.py`. Completed artifacts are resumed only
when their recorded input identities match. The original pilot used the legacy
strict judgment rule and omitted the end token; its reports remain unchanged.

This adapts the paper to Alpaca-1k and the installed 135M proxy. It is not an exact
replication of the original LLaMA-7B training, pre-experience stage or evaluator.
A positive outcome is not guaranteed; all outcomes will be reported.

Protocol reference: [Li et al., sections 3–4](https://arxiv.org/html/2308.12032v3).
