import json
from pathlib import Path

from core.geometry import compute_all_angles


# -- threshold loader --

def load_thresholds(path: str) -> dict:
    """Load threshold config from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


class SquatAnalyser:
    """
    Rule-based squat analyser for Trimester 1.
    Responsibilities: phase detection, hard constraint checking, rep counting.
    This layer is immutable — no ML, no LLM, no external calls.
    """

    THRESHOLD_PATH = Path(__file__).resolve().parent / "thresholds" / "squat_t1.json"

    def __init__(self):
        self._thresholds  = load_thresholds(self.THRESHOLD_PATH)
        self._constraints = self._thresholds["hard_constraints"]
        self._phases      = self._thresholds["phase_definitions"]
        self._rep_rules   = self._thresholds["rep_validity"]

        # -- state machine variables --
        self._current_phase  = None
        self._phase_seq      = []
        self._rep_count      = 0
        self._frames_in_s3   = 0
        self._incorrect_reps = 0

    # -- phase detection --

    def _detect_phase(self, knee_avg: float) -> str | None:
        """Map knee angle to squat phase. Returns None if in buffer zone."""
        for phase, definition in self._phases.items():
            low, high = definition["knee_angle_range"]
            if low <= knee_avg <= high:
                return phase
        return None

    # -- hard constraint checks --

    def _check_constraints(self, angles: dict) -> list[str]:
        """
        Check all hard constraints against current frame angles.
        Returns list of violated error codes (empty = all clear).
        """
        errors = []

        knee_avg    = angles["knee"]["average"]
        hip_angle   = angles["hip_vertical"]["angle"]
        tibia_avg   = angles["ankle_tibia"]["average"]
        spine_dev   = angles["spine_deviation"]["deviation"]

        # knee depth check
        if knee_avg > self._constraints["knee_flexion_max"]["value"]:
            errors.append("ERR_KNEE_TOO_DEEP")

        # hip lean forward check
        if hip_angle > self._constraints["hip_forward_lean_max"]["value"]:
            errors.append("ERR_HIP_LEAN_FORWARD")

        # hip lean backward check
        if hip_angle < self._constraints["hip_backward_lean_min"]["value"]:
            errors.append("ERR_HIP_LEAN_BACK")

        # knee over toe check
        if tibia_avg > self._constraints["ankle_tibia_max"]["value"]:
            errors.append("ERR_KNEE_OVER_TOE")

        # spine neutrality check
        if spine_dev > self._constraints["spine_neutral_deviation_max"]["value"]:
            errors.append("ERR_SPINE_DEVIATION")

        return errors

    # -- rep validity check --

    def _check_rep_valid(self) -> bool:
        """
        Validate rep based on required phase sequence.
        Required: s2 -> s3 -> s2 -> s1, with min frames in s3.
        """
        required = self._rep_rules["required_sequence"]
        min_s3   = self._rep_rules["min_frames_in_s3"]

        seq_match      = self._phase_seq[-len(required):] == required
        s3_held_enough = self._frames_in_s3 >= min_s3

        return seq_match and s3_held_enough

    # -- state machine --

    def _update_state(self, phase: str, errors: list[str]):
        """Update phase sequence and rep counter based on current phase."""
        if phase is None:
            return

        # append phase only on transition, not every frame
        if not self._phase_seq or self._phase_seq[-1] != phase:
            self._phase_seq.append(phase)

        # track how long user holds s3
        if phase == "s3_squat":
            self._frames_in_s3 += 1
        else:
            if self._phase_seq and "s3_squat" in self._phase_seq:
                pass  # s3 already counted, do not reset mid-sequence

        # rep completion: triggered when returning to s1 after full sequence
        if phase == "s1_standing" and len(self._phase_seq) >= 4:
            if self._check_rep_valid():
                if errors:
                    self._incorrect_reps += 1
                else:
                    self._rep_count += 1
            # reset sequence for next rep
            self._phase_seq    = []
            self._frames_in_s3 = 0

    # -- visibility guard --

    def _all_visible(self, angles: dict) -> bool:
        """Return True only if all required landmarks are visible."""
        return all([
            angles["knee"]["visible"],
            angles["hip_vertical"]["visible"],
            angles["ankle_tibia"]["visible"],
            angles["spine_deviation"]["visible"],
        ])

    # -- main interface --

    def analyse_frame(self, landmarks) -> dict:
        """
        Single entry point — called once per frame by amora_engine.
        Returns current analysis result without modifying external state.
        """
        angles = compute_all_angles(landmarks)

        # skip frame if landmarks not reliably visible
        if not self._all_visible(angles):
            return self._make_result(angles, phase=None, errors=[], detected=False)

        phase  = self._detect_phase(angles["knee"]["average"])
        errors = self._check_constraints(angles)

        self._update_state(phase, errors)

        return self._make_result(angles, phase, errors, detected=True)

    def _make_result(self, angles: dict, phase: str | None,
                     errors: list[str], detected: bool) -> dict:
        """Package analysis result as a clean dict for upstream layers."""
        return {
            "detected":      detected,
            "phase":         phase,
            "errors":        errors,
            "rep_count":     self._rep_count,
            "incorrect_reps": self._incorrect_reps,
            "angles":        {
                "knee_avg":    angles["knee"]["average"],
                "hip_angle":   angles["hip_vertical"]["angle"],
                "tibia_avg":   angles["ankle_tibia"]["average"],
                "spine_dev":   angles["spine_deviation"]["deviation"],
            },
        }

    def reset(self):
        """Reset all state — call at start of new session."""
        self._current_phase  = None
        self._phase_seq      = []
        self._rep_count      = 0
        self._frames_in_s3   = 0
        self._incorrect_reps = 0