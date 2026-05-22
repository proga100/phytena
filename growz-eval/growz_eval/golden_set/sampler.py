"""Stratified sampling across crop x tier strata."""

from __future__ import annotations

import random
from dataclasses import dataclass

from growz_eval.config import CLASS_BALANCE, CROP_DISTRIBUTION
from growz_eval.golden_set.models import ImageCandidate
from growz_eval.golden_set.scanner import CandidatePool
from growz_eval.logging_utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _Quota:
    crop: str
    tier: str
    source_tier: str  # which pool to draw from (ambiguous draws from rare)
    target: int


class StratifiedSampler:
    """Samples a balanced subset matching CROP_DISTRIBUTION × CLASS_BALANCE."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def sample(self, pool: CandidatePool, size: int) -> list[ImageCandidate]:
        quotas = self._compute_quotas(size)
        picked: list[ImageCandidate] = []
        for quota in quotas:
            candidates = pool.get(quota.crop, {}).get(quota.source_tier, [])
            if not candidates:
                log.warning("  no %s for %s (wanted %d)",
                            quota.source_tier, quota.crop, quota.target)
                continue
            take = min(quota.target, len(candidates))
            picked.extend(self._rng.sample(candidates, take))
        return picked

    @staticmethod
    def _compute_quotas(size: int) -> list[_Quota]:
        quotas: list[_Quota] = []
        for crop, crop_share in CROP_DISTRIBUTION.items():
            crop_n = int(round(size * crop_share))
            for tier, tier_share in CLASS_BALANCE.items():
                target = int(round(crop_n * tier_share))
                source_tier = "rare" if tier == "ambiguous" else tier
                quotas.append(_Quota(crop=crop, tier=tier,
                                     source_tier=source_tier, target=target))
        return quotas
