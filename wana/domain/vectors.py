import math


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Finite cosine with zero-vector similarity defined as zero."""
    if not left or len(left) != len(right) or not all(math.isfinite(x) for x in left + right):
        raise ValueError("embeddings must have equal nonzero dimensions and finite values")
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return (
        max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True)) / denominator))
        if denominator
        else 0.0
    )
