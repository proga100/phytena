"""Pure decision logic that combines Gemini + Claude verdicts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from growz_eval.verification.models import ClaudeVerdict, GeminiVerdict

FinalVerdict = Literal["verified", "relabeled", "ambiguous", "dataset_wrong", "error"]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True, slots=True)
class FinalDecision:
    verdict: FinalVerdict
    disease: str
    confidence: Confidence


class FinalDecisionEngine:
    """Combines reviewer verdicts into a final decision.

    Mirrors the table in README.md. No I/O — fully unit-testable.
    """

    def decide(
        self,
        *,
        dataset_label: str,
        gemini: GeminiVerdict,
        claude: ClaudeVerdict | None,
    ) -> FinalDecision:
        if gemini.verdict == "ERROR":
            return FinalDecision("error", dataset_label, "low")

        if gemini.verdict == "agree":
            return FinalDecision("verified", dataset_label, "high")

        if gemini.verdict == "uncertain":
            return FinalDecision("ambiguous", dataset_label, "low")

        # gemini disagreed — need Claude
        if claude is None or claude.verdict == "":
            return FinalDecision("ambiguous", dataset_label, "low")

        if claude.verdict == "ERROR":
            return FinalDecision("error", dataset_label, "low")
        if claude.verdict == "dataset_correct":
            return FinalDecision("verified", dataset_label, "high")
        if claude.verdict == "gemini_correct":
            return FinalDecision("relabeled", gemini.proposed_disease, "medium")
        if claude.verdict == "neither_correct":
            return FinalDecision("dataset_wrong", claude.proposed_disease, "medium")
        if claude.verdict == "uncertain":
            return FinalDecision("ambiguous", dataset_label, "low")

        return FinalDecision("error", dataset_label, "low")
