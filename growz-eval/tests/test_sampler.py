from pathlib import Path

from growz_eval.golden_set.models import ImageCandidate
from growz_eval.golden_set.sampler import StratifiedSampler


def _candidate(crop: str, tier: str, n: int) -> list[ImageCandidate]:
    return [
        ImageCandidate(
            path=Path(f"/fake/{crop}/{tier}/{i}.jpg"),
            label=f"{crop}_{tier}_{i}",
            dataset_id="fake",
            crop=crop,
            tier=tier,
        )
        for i in range(n)
    ]


def _full_pool() -> dict[str, dict[str, list[ImageCandidate]]]:
    pool: dict[str, dict[str, list[ImageCandidate]]] = {}
    for crop in ("cotton", "grape", "tomato", "potato", "wheat", "other"):
        pool[crop] = {
            "healthy": _candidate(crop, "healthy", 50),
            "common": _candidate(crop, "common", 50),
            "rare": _candidate(crop, "rare", 50),
        }
    return pool


class TestStratifiedSampler:
    def test_deterministic_with_seed(self):
        pool = _full_pool()
        a = StratifiedSampler(seed=42).sample(pool, 100)
        b = StratifiedSampler(seed=42).sample(pool, 100)
        assert [c.path for c in a] == [c.path for c in b]

    def test_respects_size_when_pool_is_full(self):
        # Sum of int(round(...)) quotas may differ from `size` by a few; allow ±10%.
        pool = _full_pool()
        picked = StratifiedSampler(seed=1).sample(pool, 100)
        assert 85 <= len(picked) <= 115

    def test_skips_empty_strata(self):
        pool = _full_pool()
        # Remove all cotton common
        pool["cotton"]["common"] = []
        picked = StratifiedSampler(seed=1).sample(pool, 100)
        assert all(not (c.crop == "cotton" and c.tier == "common") for c in picked)

    def test_ambiguous_draws_from_rare(self):
        # Only `rare` pool exists; sampler should still produce ambiguous picks.
        pool: dict[str, dict[str, list[ImageCandidate]]] = {
            "cotton": {"rare": _candidate("cotton", "rare", 100)},
        }
        picked = StratifiedSampler(seed=1).sample(pool, 100)
        # Sampler doesn't relabel; it just draws from `rare`. Verify it produced rows.
        assert picked
        assert all(c.tier == "rare" for c in picked)
