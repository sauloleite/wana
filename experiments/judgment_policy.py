"""Counterbalanced judgments: explicit legacy and paper aggregation policies."""


def combine_orders(first: str, second: str, *, baseline: str, policy: str) -> str:
    if first not in ("selected", baseline, "tie") or second not in ("selected", baseline, "tie"):
        raise ValueError("invalid order decision")
    if policy not in ("strict", "paper"):
        raise ValueError("unknown order policy")
    if first == second:
        return first
    if policy == "paper":
        if first == "tie":
            return second
        if second == "tie":
            return first
    return "tie"
