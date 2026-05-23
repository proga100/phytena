from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class RankedCandidate:
    chunk_id: str
    rank: int
    score: float | None = None


@dataclass(frozen=True)
class MergedCandidate:
    chunk_id: str
    rrf_score: float
    ranks: dict[str, int]
    scores: dict[str, float | None]


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RankedCandidate]],
    *,
    k: int = 60,
) -> list[MergedCandidate]:
    totals: dict[str, float] = defaultdict(float)
    ranks_by_chunk: dict[str, dict[str, int]] = defaultdict(dict)
    scores_by_chunk: dict[str, dict[str, float | None]] = defaultdict(dict)

    for source_name, candidates in ranked_lists.items():
        for candidate in candidates:
            totals[candidate.chunk_id] += 1.0 / (k + candidate.rank)
            ranks_by_chunk[candidate.chunk_id][source_name] = candidate.rank
            scores_by_chunk[candidate.chunk_id][source_name] = candidate.score

    merged = [
        MergedCandidate(
            chunk_id=chunk_id,
            rrf_score=score,
            ranks=ranks_by_chunk[chunk_id],
            scores=scores_by_chunk[chunk_id],
        )
        for chunk_id, score in totals.items()
    ]
    return sorted(merged, key=lambda candidate: candidate.rrf_score, reverse=True)
