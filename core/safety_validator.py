from core.squat_analyser import SquatAnalyser


class SafetyValidator:
    """
    Gatekeeper layer between ML output and feedback engine.
    Ensures ML verdict never contradicts rule-based hard constraints.
    All overrides are logged for audit trail.
    """

    def __init__(self, analyser: SquatAnalyser):
        self._analyser     = analyser
        self._override_log = []

    # -- validation --

    def validate(self, rule_result: dict, ml_quality_score: float | None = None) -> dict:
        """
        Main entry point — called once per rep by amora_engine.
        Accepts rule_result from SquatAnalyser and optional ML quality score.
        Returns final validated verdict.
        """
        errors      = rule_result["errors"]
        rep_valid   = len(errors) == 0

        # if rule-based detected violations, ML score is overridden unconditionally
        if errors:
            final_score = self._override_ml(ml_quality_score, errors)
            return self._make_verdict(
                rule_result=rule_result,
                final_score=final_score,
                overridden=True,
            )

        # if no violations, trust ML score if available
        final_score = ml_quality_score if ml_quality_score is not None else 1.0

        return self._make_verdict(
            rule_result=rule_result,
            final_score=final_score,
            overridden=False,
        )

    # -- override logic --

    def _override_ml(self, ml_score: float | None, errors: list[str]) -> float:
        """
        Force quality score to 0.0 when hard constraints are violated.
        Logs every override for audit trail.
        """
        self._log_override(ml_score, errors)
        return 0.0

    # -- audit trail --

    def _log_override(self, ml_score: float | None, errors: list[str]):
        """Record every instance where ML output was overridden by rule-based."""
        entry = {
            "ml_score_rejected": ml_score,
            "reason":            errors,
        }
        self._override_log.append(entry)

    def get_override_log(self) -> list[dict]:
        """Return full override log — used for debugging and audit."""
        return self._override_log

    def clear_override_log(self):
        """Clear log at start of new session."""
        self._override_log = []

    # -- verdict packaging --

    def _make_verdict(self, rule_result: dict,
                      final_score: float, overridden: bool) -> dict:
        """Package final validated verdict for feedback engine."""
        return {
            "phase":          rule_result["phase"],
            "errors":         rule_result["errors"],
            "rep_count":      rule_result["rep_count"],
            "incorrect_reps": rule_result["incorrect_reps"],
            "angles":         rule_result["angles"],
            "quality_score":  round(final_score, 4),
            "ml_overridden":  overridden,
            "rep_safe":       len(rule_result["errors"]) == 0,
        }

    def reset(self):
        """Reset state — call at start of new session."""
        self.clear_override_log()