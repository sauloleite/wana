from wana.domain.contamination import Level, Report


def render(report: Report) -> str:
    counts = ", ".join(
        f"{level.value}={sum(h.level == level for h in report.hits)}"
        for level in (Level.EXACT, Level.NEAR)
    )
    status = "PASS" if report.ok else "FAIL"
    return (
        f"{status}: train={report.train_records}, eval={report.eval_records}; {counts}. "
        "SEMANTIC=not checked"
    )
