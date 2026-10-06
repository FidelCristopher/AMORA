import cv2
import mediapipe as mp

from pipeline.amora_engine import AmoraEngine


# -- mediapipe setup --
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def draw_overlay(frame, verdict: dict):
    """Render verdict and posture feedback onto the camera frame."""
    h, w = frame.shape[:2]

    # Status / Phase & Rep Counter
    phase = verdict["phase"] or "detecting..."
    cv2.putText(frame, f"Phase: {phase}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Correct: {verdict['rep_count']}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2)
    cv2.putText(frame, f"Incorrect: {verdict['incorrect_reps']}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    # Angle telemetry display
    knee_ang = verdict['angles']['knee_avg']
    hip_ang = verdict['angles']['hip_angle']
    ankle_ang = verdict['angles']['tibia_avg']

    cv2.putText(frame, f"Knee Vert: {knee_ang:.1f} deg", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    cv2.putText(frame, f"Hip Vert:  {hip_ang:.1f} deg", (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    cv2.putText(frame, f"Shin Vert: {ankle_ang:.1f} deg", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

    # Camera Alignment Warning
    if not verdict.get("camera_aligned", True):
        cv2.putText(frame, "CAMERA: PLEASE FACE SIDEWAYS", (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
    elif verdict.get("lower_hips"):
        cv2.putText(frame, "LOWER YOUR HIPS", (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    elif not verdict["errors"]:
        cv2.putText(frame, "FORM OK", (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Error / form warnings display
    if verdict["errors"]:
        for i, error in enumerate(verdict["errors"]):
            cv2.putText(frame, error, (20, 275 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Active side & Offset telemetry
    side = verdict.get("side", "unknown")
    offset = verdict.get("offset_angle", 0.0)
    cv2.putText(frame, f"Side: {side.upper()} | Offset: {offset:.1f}", (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Quality score
    score = verdict["quality_score"]
    cv2.putText(frame, f"Quality Score: {score:.2f}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def main():
    engine = AmoraEngine()
    engine.start_session()

    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # draw MediaPipe skeleton overlay
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                # process frame through engine
                h, w = frame.shape[:2]
                verdict = engine.process_frame(results.pose_landmarks.landmark, w, h)
                draw_overlay(frame, verdict)

                # debug logging ke terminal
                status_str = "ALIGN_OK" if verdict.get("camera_aligned", True) else "WARN_ALIGN"
                print(
                    f"phase={verdict['phase']} | "
                    f"knee={verdict['angles']['knee_avg']:.1f} | "
                    f"hip={verdict['angles']['hip_angle']:.1f} | "
                    f"tibia={verdict['angles']['tibia_avg']:.1f} | "
                    f"offset={verdict.get('offset_angle', 0.0):.1f} [{status_str}] | "
                    f"correct={verdict['rep_count']} | "
                    f"errors={verdict['errors']}"
                )

            cv2.imshow("AMORA — Squat Analyser T1", frame)

            # press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
