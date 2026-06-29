from __future__ import annotations

from sophons.memory.long_term.search import RetrievalResult

# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

DEFAULT_RRF_CONSTANT = 60


def reciprocal_rank_score(
    rank: int,
    constant: int = DEFAULT_RRF_CONSTANT,
) -> float:
    """
    Compute the Reciprocal Rank Fusion (RRF) score for a given rank.

    RRF is used to combine results from multiple retrievers without needing
    to normalise their raw scores.  A lower rank (closer to 1) gives a
    higher score.

    Formula: 1 / (constant + rank)
    """
    return 1 / (constant + rank)


def fuse_results(
    result_lists: list[list[RetrievalResult]],
    constant: int = DEFAULT_RRF_CONSTANT,
) -> list[RetrievalResult]:
    """
    Merge multiple ranked result lists into one using Reciprocal Rank Fusion.

    Each entry_id accumulates RRF scores from every list it appears in.
    The fused list is sorted by total score descending.

    Args:
        result_lists: One list per retriever, each sorted by relevance.
        constant:     The RRF constant k. Higher values reduce the impact of
                      top-ranked results. Default 60 is the standard value.

    Returns:
        A single merged list sorted by fused score descending.
    """
    scores: dict[str, float] = {}
    best: dict[str, RetrievalResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            rrf = reciprocal_rank_score(rank, constant)
            scores[result.entry_id] = scores.get(result.entry_id, 0.0) + rrf
            if result.entry_id not in best or result.score > best[result.entry_id].score:
                best[result.entry_id] = result

    from dataclasses import replace

    return sorted(
        [replace(best[eid], score=scores[eid]) for eid in best],
        key=lambda r: r.score,
        reverse=True,
    )
