"""Orchestrates the two-stage verification pipeline."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from growz_eval.io_csv.repo import VerifiedRow
from growz_eval.logging_utils import get_logger, step_banner
from growz_eval.verification.decision import FinalDecisionEngine
from growz_eval.verification.models import ClaudeVerdict, GeminiVerdict
from growz_eval.verification.reviewers import ClaudeReviewer, GeminiReviewer

log = get_logger(__name__)


class VerificationPipeline:
    def __init__(
        self,
        *,
        gemini: GeminiReviewer,
        claude: ClaudeReviewer | None,
        sleep_seconds: float,
        decision_engine: FinalDecisionEngine | None = None,
    ) -> None:
        self._gemini = gemini
        self._claude = claude
        self._sleep = sleep_seconds
        self._engine = decision_engine or FinalDecisionEngine()

    def run(self, rows: list[VerifiedRow], skip_ids: Iterable[str] = ()) -> list[VerifiedRow]:
        skip = set(skip_ids)
        if skip:
            log.info("[RESUME] %d rows already processed, skipping them", len(skip))

        active = [r for r in rows if r.id not in skip]
        gemini_by_id = self._run_gemini_stage(active)

        disputes = [r for r in active if gemini_by_id[r.id].verdict == "disagree"]
        claude_by_id = self._run_claude_stage(disputes) if self._claude else {}

        for row in active:
            g = gemini_by_id[row.id]
            c = claude_by_id.get(row.id)
            self._apply_verdicts(row, g, c)

        return active

    def _run_gemini_stage(self, rows: list[VerifiedRow]) -> dict[str, GeminiVerdict]:
        step_banner(log, f"STAGE 1: Gemini on {len(rows)} images")
        results: dict[str, GeminiVerdict] = {}
        for i, row in enumerate(rows, 1):
            image_path = Path(row.image_path)
            if not image_path.exists():
                log.warning("[%d/%d] SKIP — missing %s", i, len(rows), image_path)
                results[row.id] = GeminiVerdict(verdict="ERROR", reasoning="missing file")
                continue

            verdict = self._gemini.review(image_path, row.crop, row.true_disease)
            results[row.id] = verdict
            self._log_gemini_row(i, len(rows), row, verdict)
            time.sleep(self._sleep)

        counts = Counter(v.verdict for v in results.values())
        log.info("Stage 1 results:")
        for kind, n in sorted(counts.items()):
            log.info("  %-15s %d", kind, n)
        return results

    def _run_claude_stage(self, rows: list[VerifiedRow]) -> dict[str, ClaudeVerdict]:
        assert self._claude is not None
        if not rows:
            log.info("No disputes — Claude stage skipped.")
            return {}

        step_banner(log, f"STAGE 2: Claude tiebreak on {len(rows)} disputes")
        results: dict[str, ClaudeVerdict] = {}
        for i, row in enumerate(rows, 1):
            verdict = self._claude.review(
                Path(row.image_path),
                row.crop,
                row.true_disease,
                row.gemini_proposed or "",
                row.gemini_reasoning or "",
            )
            results[row.id] = verdict
            log.info(
                "[%3d/%d] %-8s dataset=%-15s gemini=%-15s -> claude: %s",
                i, len(rows), row.crop,
                row.true_disease[:15], (row.gemini_proposed or "")[:15],
                verdict.verdict,
            )
            time.sleep(self._sleep)
        return results

    def _apply_verdicts(
        self,
        row: VerifiedRow,
        gemini: GeminiVerdict,
        claude: ClaudeVerdict | None,
    ) -> None:
        row.gemini_verdict = gemini.verdict
        row.gemini_proposed = gemini.proposed_disease
        row.gemini_reasoning = gemini.reasoning
        row.photo_quality = gemini.photo_quality or row.photo_quality

        if claude is not None:
            row.claude_verdict = claude.verdict
            row.claude_proposed = claude.proposed_disease
            row.claude_reasoning = claude.reasoning
            row.claude_confidence = claude.confidence

        decision = self._engine.decide(
            dataset_label=row.true_disease,
            gemini=gemini,
            claude=claude,
        )
        row.final_verdict = decision.verdict
        row.final_disease = decision.disease
        row.final_confidence = decision.confidence

    @staticmethod
    def _log_gemini_row(i: int, total: int, row: VerifiedRow, verdict: GeminiVerdict) -> None:
        marker = {
            "agree": "OK", "disagree": "??", "uncertain": "??", "ERROR": "!!",
        }.get(verdict.verdict, "?")
        suffix = f"| {verdict.proposed_disease[:25]}" if verdict.verdict == "disagree" else ""
        log.info(
            "[%3d/%d] %s %-8s label=%-20s -> %-10s %s",
            i, total, marker, row.crop, row.true_disease[:20], verdict.verdict, suffix,
        )
