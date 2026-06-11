from core.squat_analyser import SquatAnalyser
from core.safety_validator import SafetyValidator


class AmoraEngine:
    """
    Main pipeline orchestrator.
    Execution order per frame: geometry -> rule-based -> safety validator -> result.
    ML and LLM layers will be plugged in here at Phase 3 and Phase 5.
    """

    def __init__(self):
        self._analyser  = SquatAnalyser()
        self._validator = SafetyValidator(self._analyser)

    # -- per-frame entry point --

    def process_frame(self, landmarks) -> dict:
        """
        Called once per frame from run_realtime.py.
        Returns validated verdict ready for UI rendering.
        """
        # Layer 1 + 2: geometry and rule-based analysis
        rule_result = self._analyser.analyse_frame(landmarks)

        # Layer 4: safety validation
        # ml_quality_score is None until Phase 3 ML model is integrated
        verdict = self._validator.validate(
            rule_result=rule_result,
            ml_quality_score=None,
        )

        return verdict

    # -- session controls --

    def start_session(self):
        """Reset all state — call before every new exercise session."""
        self._analyser.reset()
        self._validator.reset()

    def get_override_log(self) -> list[dict]:
        """Expose override log for debugging and audit trail."""
        return self._validator.get_override_log()