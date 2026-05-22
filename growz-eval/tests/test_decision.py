from growz_eval.verification.decision import FinalDecisionEngine
from growz_eval.verification.models import ClaudeVerdict, GeminiVerdict


class TestFinalDecisionEngine:
    engine = FinalDecisionEngine()

    def _decide(self, gv: GeminiVerdict, cv: ClaudeVerdict | None = None, label: str = "x"):
        return self.engine.decide(dataset_label=label, gemini=gv, claude=cv)

    def test_gemini_agree_is_verified(self):
        d = self._decide(GeminiVerdict(verdict="agree"))
        assert d.verdict == "verified"
        assert d.disease == "x"
        assert d.confidence == "high"

    def test_gemini_uncertain_is_ambiguous(self):
        d = self._decide(GeminiVerdict(verdict="uncertain"))
        assert d.verdict == "ambiguous"
        assert d.confidence == "low"

    def test_gemini_error_is_error(self):
        d = self._decide(GeminiVerdict(verdict="ERROR"))
        assert d.verdict == "error"

    def test_disagree_no_claude_is_ambiguous(self):
        d = self._decide(GeminiVerdict(verdict="disagree", proposed_disease="y"))
        assert d.verdict == "ambiguous"

    def test_disagree_claude_says_dataset_correct(self):
        d = self._decide(
            GeminiVerdict(verdict="disagree", proposed_disease="y"),
            ClaudeVerdict(verdict="dataset_correct"),
        )
        assert d.verdict == "verified"
        assert d.disease == "x"
        assert d.confidence == "high"

    def test_disagree_claude_says_gemini_correct(self):
        d = self._decide(
            GeminiVerdict(verdict="disagree", proposed_disease="y"),
            ClaudeVerdict(verdict="gemini_correct"),
        )
        assert d.verdict == "relabeled"
        assert d.disease == "y"
        assert d.confidence == "medium"

    def test_disagree_claude_says_neither(self):
        d = self._decide(
            GeminiVerdict(verdict="disagree", proposed_disease="y"),
            ClaudeVerdict(verdict="neither_correct", proposed_disease="z"),
        )
        assert d.verdict == "dataset_wrong"
        assert d.disease == "z"

    def test_disagree_claude_uncertain_is_ambiguous(self):
        d = self._decide(
            GeminiVerdict(verdict="disagree"),
            ClaudeVerdict(verdict="uncertain"),
        )
        assert d.verdict == "ambiguous"

    def test_disagree_claude_error_propagates(self):
        d = self._decide(
            GeminiVerdict(verdict="disagree"),
            ClaudeVerdict(verdict="ERROR"),
        )
        assert d.verdict == "error"
