import cv2
import mediapipe as mp

from pipeline.amora_engine import AmoraEngine


# -- mediapipe setup --
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def draw_overlay(frame, verdict: dict):
    """Render verdict information onto the camera frame."""
    h, w = frame.shape[:2]

    # phase and rep counter
    phase = verdict["phase"] or "detecting..."
    cv2.putText(frame, f"Phase: {phase}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {verdict['rep_count']}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Incorrect: {verdict['incorrect_reps']}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    # error display
    if verdict["errors"]:
        for i, error in enumerate(verdict["errors"]):
            cv2.putText(frame, error, (10, 150 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "FORM OK", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # quality score
    score = verdict["quality_score"]
    cv2.putText(frame, f"Quality: {score:.2f}", (10, h - 20),
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
            results   = pose.process(rgb_frame)

            if results.pose_landmarks:
                # draw MediaPipe skeleton overlay
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                # process frame through engine
                verdict = engine.process_frame(results.pose_landmarks.landmark)
                draw_overlay(frame, verdict)

            cv2.imshow("AMORA — Squat Analyser T1", frame)

            # press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()