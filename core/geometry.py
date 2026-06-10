import numpy as np


# -- vector & angle utilities --

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle at point b formed by vectors b->a and b->c."""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def calculate_vertical_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate angle between vector a->b and vertical axis (downward)."""
    ab = b - a
    vertical = np.array([0, 1, 0])
    cosine = np.dot(ab, vertical) / (np.linalg.norm(ab) * np.linalg.norm(vertical) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return midpoint between two landmarks."""
    return (a + b) / 2.0


# -- landmark extraction --

def extract_landmark(landmarks, index: int) -> np.ndarray:
    """Extract x, y, z from a single MediaPipe landmark as numpy array."""
    lm = landmarks[index]
    return np.array([lm.x, lm.y, lm.z])


def get_visibility(landmarks, indices: list[int]) -> bool:
    """Return True if all specified landmarks are visible above threshold."""
    return all(landmarks[i].visibility > 0.5 for i in indices)


# -- angle calculations (bilateral: average left & right) --

def get_knee_angle(landmarks) -> dict:
    """
    Knee flexion angle — average of left and right.
    Landmarks: hip(23/24), knee(25/26), ankle(27/28)
    """
    # left side
    left_hip   = extract_landmark(landmarks, 23)
    left_knee  = extract_landmark(landmarks, 25)
    left_ankle = extract_landmark(landmarks, 27)

    # right side
    right_hip   = extract_landmark(landmarks, 24)
    right_knee  = extract_landmark(landmarks, 26)
    right_ankle = extract_landmark(landmarks, 28)

    left_angle  = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)

    visible = get_visibility(landmarks, [23, 24, 25, 26, 27, 28])

    return {
        "left":    round(left_angle, 2),
        "right":   round(right_angle, 2),
        "average": round((left_angle + right_angle) / 2.0, 2),
        "visible": visible,
    }


def get_hip_vertical_angle(landmarks) -> dict:
    """
    Hip forward/backward lean — angle from shoulder_mid to hip_mid against vertical.
    Landmarks: shoulder(11/12), hip(23/24)
    """
    # left side
    left_shoulder = extract_landmark(landmarks, 11)
    left_hip      = extract_landmark(landmarks, 23)

    # right side
    right_shoulder = extract_landmark(landmarks, 12)
    right_hip      = extract_landmark(landmarks, 24)

    # midpoints represent body centerline
    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid      = midpoint(left_hip, right_hip)

    angle   = calculate_vertical_angle(shoulder_mid, hip_mid)
    visible = get_visibility(landmarks, [11, 12, 23, 24])

    return {
        "angle":   round(angle, 2),
        "visible": visible,
    }


def get_ankle_tibia_angle(landmarks) -> dict:
    """
    Ankle/tibia angle (knee-over-toe check) — average of left and right.
    Landmarks: knee(25/26), ankle(27/28), foot_index(31/32)
    """
    # left side
    left_knee       = extract_landmark(landmarks, 25)
    left_ankle      = extract_landmark(landmarks, 27)
    left_foot_index = extract_landmark(landmarks, 31)

    # right side
    right_knee       = extract_landmark(landmarks, 26)
    right_ankle      = extract_landmark(landmarks, 28)
    right_foot_index = extract_landmark(landmarks, 32)

    left_angle  = calculate_angle(left_knee, left_ankle, left_foot_index)
    right_angle = calculate_angle(right_knee, right_ankle, right_foot_index)

    visible = get_visibility(landmarks, [25, 26, 27, 28, 31, 32])

    return {
        "left":    round(left_angle, 2),
        "right":   round(right_angle, 2),
        "average": round((left_angle + right_angle) / 2.0, 2),
        "visible": visible,
    }


def get_spine_deviation(landmarks) -> dict:
    """
    Spine neutrality — lateral deviation of shoulder_mid from hip_mid against vertical.
    Landmarks: shoulder(11/12), hip(23/24)
    """
    left_shoulder  = extract_landmark(landmarks, 11)
    right_shoulder = extract_landmark(landmarks, 12)
    left_hip       = extract_landmark(landmarks, 23)
    right_hip      = extract_landmark(landmarks, 24)

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid      = midpoint(left_hip, right_hip)

    # deviation measured as angle between spine vector and vertical
    deviation = calculate_vertical_angle(hip_mid, shoulder_mid)
    visible   = get_visibility(landmarks, [11, 12, 23, 24])

    return {
        "deviation": round(deviation, 2),
        "visible":   visible,
    }


# -- main interface --

def compute_all_angles(landmarks) -> dict:
    """
    Single entry point — called once per frame by squat_analyser.
    Returns all angles needed for constraint checking.
    """
    return {
        "knee":           get_knee_angle(landmarks),
        "hip_vertical":   get_hip_vertical_angle(landmarks),
        "ankle_tibia":    get_ankle_tibia_angle(landmarks),
        "spine_deviation": get_spine_deviation(landmarks),
    }