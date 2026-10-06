import json
import time
from pathlib import Path


class RepDataCollector:
    """
    Automated kinematic time-series data collector for squat repetitions.
    Captures per-frame angles from start of descent (s2) through bottom (s3)
    until return to stand (s1). Saves standardized repetitions into ml/dataset/.
    """

    DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "dataset"

    def __init__(self, output_dir: Path | str | None = None):
        self.output_dir = Path(output_dir) if output_dir else self.DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._current_rep_frames = []
        self._is_recording = False
        self._rep_index = 0
        self._recorded_errors = set()

    def process_frame(self, verdict: dict):
        """
        Observes each frame from AmoraEngine verdict.
        Records time-series data when a rep starts, and persists when rep completes.
        """
        phase = verdict.get("phase")
        angles = verdict.get("angles", {})
        errors = verdict.get("errors", [])

        # Start recording when movement begins (entering s2 transition)
        if phase == "s2_transition" and not self._is_recording:
            self._is_recording = True
            self._current_rep_frames = []
            self._recorded_errors = set()

        if self._is_recording:
            # Collect current frame kinematics
            frame_data = {
                "timestamp": round(time.time(), 4),
                "phase": phase,
                "knee_angle": angles.get("knee_avg", 0.0),
                "hip_angle": angles.get("hip_angle", 0.0),
                "tibia_angle": angles.get("tibia_avg", 0.0),
                "spine_dev": angles.get("spine_dev", 0.0),
            }
            self._current_rep_frames.append(frame_data)

            for err in errors:
                self._recorded_errors.add(err)

            # Movement completed: user returned to standing (s1)
            if phase == "s1_standing" and len(self._current_rep_frames) > 5:
                self._save_rep(verdict)
                self._is_recording = False
                self._current_rep_frames = []
                self._recorded_errors = set()

    def _save_rep(self, verdict: dict):
        """Save captured sequence to a JSON file labeled with correctness."""
        if len(self._current_rep_frames) < 10:
            # Ignore false triggers / too short noise
            return

        self._rep_index += 1
        is_safe = len(self._recorded_errors) == 0
        label = "correct" if is_safe else "incorrect"

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"rep_{timestamp_str}_{self._rep_index:03d}_{label}.json"
        filepath = self.output_dir / filename

        payload = {
            "rep_id": self._rep_index,
            "timestamp": timestamp_str,
            "label": label,
            "safe": is_safe,
            "errors": list(self._recorded_errors),
            "frame_count": len(self._current_rep_frames),
            "trajectory": self._current_rep_frames,
        }

        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"\n[DATA COLLECTOR] Saved {label.upper()} rep #{self._rep_index} ({len(self._current_rep_frames)} frames) -> {filepath.name}")

    def get_summary(self) -> dict:
        """Returns statistics of collected dataset."""
        files = list(self.output_dir.glob("*.json"))
        correct_count = len([f for f in files if "correct" in f.name and "incorrect" not in f.name])
        incorrect_count = len([f for f in files if "incorrect" in f.name])
        return {
            "total_files": len(files),
            "correct": correct_count,
            "incorrect": incorrect_count,
            "directory": str(self.output_dir),
        }
