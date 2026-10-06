import json
import time
from pathlib import Path

from core.geometry import compute_all_angles


def load_thresholds(path: str) -> dict:
    """Load threshold config from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


class SquatAnalyser:
    """
    Rule-based squat analyser for AMORA (adapted with reference kinematics).
    Responsibilities:
      - Camera alignment validation (offset angle)
      - Phase/State detection (s1_standing, s2_transition, s3_squat)
      - Constraint & posture error checking
      - Rep counting & sequence validation
      - Inactivity monitoring & timeout resets
    """

    THRESHOLD_PATH = Path(__file__).resolve().parent / "thresholds" / "squat_t1.json"

    def __init__(self):
        self._thresholds = load_thresholds(self.THRESHOLD_PATH)
        self._constraints = self._thresholds["hard_constraints"]
        self._phases = self._thresholds["phase_definitions"]
        self._camera_cfg = self._thresholds.get(
            "camera_alignment", {"offset_threshold": 45.0, "inactive_threshold": 15.0}
        )

        # -- rep & state tracking --
        self._phase_seq = []
        self._rep_count = 0
        self._incorrect_reps = 0
        self._incorrect_posture = False
        self._lower_hips = False
        self._camera_aligned = True

        # Smoothing buffer for camera offset to avoid 1-frame jitter
        self._offset_history = []

        # -- inactivity tracking --
        self._prev_state = None
        self._curr_state = None
        self._start_inactive_time = time.perf_counter()
        self._inactive_time = 0.0

    # -- state / phase detection --

    def _detect_phase(self, knee_vertical_angle: float) -> str | None:
        """Map knee vertical angle to squat phase (s1, s2, s3)."""
        for phase_name, definition in self._phases.items():
            low, high = definition["knee_angle_range"]
            if low <= knee_vertical_angle <= high:
                return phase_name
        return None

    # -- camera alignment check with smoothing --

    def _update_camera_alignment(self, offset_angle: float) -> bool:
        """
        Smooth offset angle across recent frames to avoid temporary false-positives.
        Returns True if camera is properly aligned (side-view).
        """
        self._offset_history.append(offset_angle)
        if len(self._offset_history) > 5:
            self._offset_history.pop(0)

        avg_offset = sum(self._offset_history) / len(self._offset_history)
        # In reference: side view has offset <= threshold (typically <= 35-45)
        self._camera_aligned = avg_offset <= self._camera_cfg["offset_threshold"]
        return self._camera_aligned

    # -- error & constraint checks --

    def _check_constraints(self, angles: dict, phase: str | None) -> list[str]:
        """
        Evaluate frame against kinematic constraints.
        Posture errors only penalize repetitions; camera alignment is a separate warning.
        """
        errors = []

        knee_angle = angles["knee"]["angle"]
        hip_angle = angles["hip_vertical"]["angle"]
        ankle_angle = angles["ankle_tibia"]["angle"]

        # 1. Lower hips cue (informative coaching cue during transition descent)
        if 50.0 <= knee_angle <= 70.0 and self._phase_seq.count("s2") == 1:
            self._lower_hips = True
        else:
            self._lower_hips = False

        # 2. Squat too deep (> 96 degrees)
        if knee_angle > self._constraints["knee_flexion_max"]["value"]:
            errors.append("ERR_KNEE_TOO_DEEP")

        # 3. Hip leaning too far forward
        if hip_angle > self._constraints["hip_forward_lean_max"]["value"]:
            errors.append("ERR_HIP_LEAN_FORWARD")

        # 4. Hip leaning backward during descent/transition
        if hip_angle < self._constraints["hip_backward_lean_min"]["value"] and self._phase_seq.count("s2") == 1:
            errors.append("ERR_HIP_LEAN_BACK")

        # 5. Knee falling over toe (excessive shin forward angle)
        if ankle_angle > self._constraints["ankle_tibia_max"]["value"]:
            errors.append("ERR_KNEE_OVER_TOE")

        return errors

    # -- state sequence update matching reference --

    def _update_state_sequence(self, phase: str):
        """
        Matches reference _update_state_sequence:
        s2 appended if going down or coming up; s3 appended once when bottom reached.
        """
        if phase == "s2_transition":
            s2_count = self._phase_seq.count("s2")
            s3_in_seq = "s3" in self._phase_seq

            if (not s3_in_seq and s2_count == 0) or (s3_in_seq and s2_count == 1):
                self._phase_seq.append("s2")

        elif phase == "s3_squat":
            if "s3" not in self._phase_seq and "s2" in self._phase_seq:
                self._phase_seq.append("s3")

    # -- rep evaluation logic --

    def _evaluate_rep(self):
        """
        Triggered when returning to standing (s1_standing).
        Valid rep requires sequence ['s2', 's3', 's2'] without errors.
        """
        if len(self._phase_seq) == 3 and not self._incorrect_posture:
            self._rep_count += 1
        elif self._incorrect_posture:
            self._incorrect_reps += 1
        elif "s2" in self._phase_seq and len(self._phase_seq) == 1:
            # Half-rep / aborted rep
            self._incorrect_reps += 1

        # Reset rep state
        self._phase_seq = []
        self._incorrect_posture = False
        self._lower_hips = False

    # -- inactivity logic --

    def _check_inactivity(self, current_phase: str | None):
        """Reset counters if user remains stationary in the same state for too long."""
        now = time.perf_counter()
        if current_phase == self._prev_state and current_phase is not None:
            self._inactive_time += (now - self._start_inactive_time)
            self._start_inactive_time = now

            if self._inactive_time >= self._camera_cfg["inactive_threshold"]:
                self.reset()
        else:
            self._start_inactive_time = now
            self._inactive_time = 0.0

        self._prev_state = current_phase

    # -- visibility guard --

    def _all_visible(self, angles: dict) -> bool:
        """Return True only if required landmarks are visible."""
        return all([
            angles["knee"]["visible"],
            angles["hip_vertical"]["visible"],
            angles["ankle_tibia"]["visible"],
        ])

    # -- main interface --

    def analyse_frame(self, landmarks, frame_w: int = 640, frame_h: int = 480) -> dict:
        """
        Single entry point — called per frame by AmoraEngine.
        """
        angles = compute_all_angles(landmarks, frame_w, frame_h)

        if not self._all_visible(angles):
            return self._make_result(angles, phase=None, errors=[], camera_aligned=True, detected=False)

        camera_aligned = self._update_camera_alignment(angles["offset_angle"])
        phase = self._detect_phase(angles["knee"]["angle"])
        self._curr_state = phase
        self._check_inactivity(phase)

        errors = self._check_constraints(angles, phase)

        # Update state sequence only if camera is reasonably aligned
        if phase is not None and camera_aligned:
            self._update_state_sequence(phase)

        # Flag posture error if violation occurs while rep is in progress
        if errors and ("s2" in self._phase_seq or phase == "s3_squat"):
            self._incorrect_posture = True

        # When returning to standing position, evaluate the completed rep
        if phase == "s1_standing":
            self._evaluate_rep()

        return self._make_result(angles, phase, errors, camera_aligned=camera_aligned, detected=True)

    def _make_result(self, angles: dict, phase: str | None, errors: list[str], camera_aligned: bool, detected: bool) -> dict:
        """Package analysis result dictionary."""
        return {
            "detected": detected,
            "phase": phase,
            "errors": errors,
            "camera_aligned": camera_aligned,
            "rep_count": self._rep_count,
            "incorrect_reps": self._incorrect_reps,
            "lower_hips": self._lower_hips,
            "side": angles.get("side", "unknown"),
            "multiplier": angles.get("multiplier", 1),
            "offset_angle": angles.get("offset_angle", 0.0),
            "angles": {
                "knee_avg": angles["knee"]["angle"],
                "hip_angle": angles["hip_vertical"]["angle"],
                "tibia_avg": angles["ankle_tibia"]["angle"],
                "spine_dev": angles["spine_deviation"]["deviation"],
            },
            "landmarks": angles.get("landmarks", {}),
        }

    def reset(self):
        """Reset session counters and state sequence."""
        self._phase_seq = []
        self._rep_count = 0
        self._incorrect_reps = 0
        self._incorrect_posture = False
        self._lower_hips = False
        self._prev_state = None
        self._curr_state = None
        self._start_inactive_time = time.perf_counter()
        self._inactive_time = 0.0
        self._offset_history = []
