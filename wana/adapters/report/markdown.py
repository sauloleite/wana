from wana.adapters.report.terminal import render
from wana.domain.contamination import Report


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("`", "&#96;")
    )


def render_markdown(report: Report) -> str:
    lines = [
        "# Contamination audit",
        "",
        render(report),
        "",
        "| Train | Evaluation | Level | Similarity | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for hit in report.hits:
        lines.append(
            "| "
            + " | ".join(
                _escape(str(value))
                for value in (hit.train_id, hit.eval_id, hit.level.value, hit.value, hit.evidence)
            )
            + " |"
        )
    return "\n".join(lines) + "\n"
