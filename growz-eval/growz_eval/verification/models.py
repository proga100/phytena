"""Domain types for verification stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GeminiVerdictKind = Literal["agree", "disagree", "uncertain", "ERROR"]
ClaudeVerdictKind = Literal[
    "dataset_correct", "gemini_correct", "neither_correct", "uncertain", "ERROR", ""
]


@dataclass(frozen=True, slots=True)
class GeminiVerdict:
    verdict: GeminiVerdictKind
    proposed_disease: str = ""
    reasoning: str = ""
    photo_quality: str = ""


@dataclass(frozen=True, slots=True)
class ClaudeVerdict:
    verdict: ClaudeVerdictKind
    proposed_disease: str = ""
    reasoning: str = ""
    confidence: str = ""


class ReviewerError(RuntimeError):
    """Raised by reviewers for hard, non-retryable failures (auth, bad config)."""
