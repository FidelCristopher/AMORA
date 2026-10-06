import numpy as np


# -- core vector math --

def find_angle(p1: np.ndarray, p2: np.ndarray, ref_pt: np.ndarray = np.array([0, 0])) -> float:
    """
    Calculate angle between vectors (p1 - ref_pt) and (p2 - ref_pt).
    Matches reference implementation in utils.py.
    """
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    norm_product = np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref)
    if norm_product < 1e-6:
        return 0.0

    cos_theta = np.dot(p1_ref, p2_ref) / norm_product
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    degree = (180.0 / np.pi) * theta
    return float(degree)


def find_vertical_angle(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate angle between vector (p1 -> p2) and the upward vertical axis.
    Reference method: find_angle(p1, np.array([p2[0], 0]), p2)
    ~0° when p1 is directly above p2, increases as it tilts.
    """
    ref_vertical = np.array([p2[0], 0])
    return find_angle(p1, ref_vertical, p2)


# -- landmark extraction --

def extract_landmark_2d(landmarks, index: int, frame_w: int, frame_h: int) -> np.ndarray:
    """
    Extract denormalized 2D pixel coordinates from MediaPipe landmark list/dict.
    Matches reference: int(lm.x * width), int(lm.y * height).
    """
    lm = landmarks[index]
    return np.array([int(lm.x * frame_w), int(lm.y * frame_h)])


def get_visibility(landmarks, indices: list) -> bool:
    """Return True if all specified landmarks have acceptable visibility."""
    return all(landmarks[i].visibility > 0.3 for i in indices)


# -- active side selection & camera orientation --

def get_active_side(landmarks, frame_w: int, frame_h: int) -> tuple[str, int]:
    """
    Select active side and multiplier for angle arcs based on reference logic.
    Compares vertical distance between foot and shoulder:
    dist_l_sh_hip = abs(left_foot[1] - left_shoulder[1])
    dist_r_sh_hip = abs(right_foot[1] - right_shoulder[1])
    Whichever is larger is closer/more visible to camera.
    Returns: (side_name, multiplier)
    """
    left_shoulder = extract_landmark_2d(landmarks, 11, frame_w, frame_h)
    left_foot     = extract_landmark_2d(landmarks, 31, frame_w, frame_h)
    right_shoulder = extract_landmark_2d(landmarks, 12, frame_w, frame_h)
    right_foot    = extract_landmark_2d(landmarks, 32, frame_w, frame_h)

    dist_left  = abs(left_foot[1] - left_shoulder[1])
    dist_right = abs(right_foot[1] - right_shoulder[1])

    if dist_left > dist_right:
        return "left", -1
    else:
        return "right", 1


def get_camera_offset_angle(landmarks, frame_w: int, frame_h: int) -> float:
    """
    Calculate camera offset angle to ensure side-view alignment.
    Reference method: find_angle(left_shldr, right_shldr, nose)
    Angle <= 35° indicates proper side view.
    """
    nose           = extract_landmark_2d(landmarks, 0, frame_w, frame_h)
    left_shoulder  = extract_landmark_2d(landmarks, 11, frame_w, frame_h)
    right_shoulder = extract_landmark_2d(landmarks, 12, frame_w, frame_h)

    return find_angle(left_shoulder, right_shoulder, nose)


# -- landmark mapping --

SIDE_INDICES = {
    "left": {
        "shoulder": 11,
        "elbow": 13,
        "wrist": 15,
        "hip": 23,
        "knee": 25,
        "ankle": 27,
        "foot": 31,
    },
    "right": {
        "shoulder": 12,
        "elbow": 14,
        "wrist": 16,
        "hip": 24,
        "knee": 26,
        "ankle": 28,
        "foot": 32,
    },
}


def extract_side_landmarks(landmarks, side: str, frame_w: int, frame_h: int) -> dict:
    """Extract 2D pixel coordinates for the active side."""
    idx = SIDE_INDICES[side]
    return {
        key: extract_landmark_2d(landmarks, idx[key], frame_w, frame_h)
        for key in idx
    }


# -- main entry point for geometry --

def compute_all_angles(landmarks, frame_w: int = 640, frame_h: int = 480) -> dict:
    """
    Single entry point — called once per frame by squat analyser.
    Uses vertical reference calculation exactly as defined in the reference repository.
    """
    side, multiplier = get_active_side(landmarks, frame_w, frame_h)
    pts = extract_side_landmarks(landmarks, side, frame_w, frame_h)
    offset_angle = get_camera_offset_angle(landmarks, frame_w, frame_h)

    # Hip vertical angle: find_angle(shoulder, [hip_x, 0], hip)
    hip_vertical_angle = find_vertical_angle(pts["shoulder"], pts["hip"])

    # Knee vertical angle: find_angle(hip, [knee_x, 0], knee)
    knee_vertical_angle = find_vertical_angle(pts["hip"], pts["knee"])

    # Ankle vertical angle: find_angle(knee, [ankle_x, 0], ankle)
    ankle_vertical_angle = find_vertical_angle(pts["knee"], pts["ankle"])

    return {
        "side": side,
        "multiplier": multiplier,
        "offset_angle": round(offset_angle, 2),
        "knee": {
            "angle": round(knee_vertical_angle, 2),
            "visible": get_visibility(landmarks, [23, 24, 25, 26]),
        },
        "hip_vertical": {
            "angle": round(hip_vertical_angle, 2),
            "visible": get_visibility(landmarks, [11, 12, 23, 24]),
        },
        "ankle_tibia": {
            "angle": round(ankle_vertical_angle, 2),
            "visible": get_visibility(landmarks, [25, 26, 27, 28]),
        },
        "spine_deviation": {
            "deviation": round(hip_vertical_angle, 2),
            "visible": get_visibility(landmarks, [11, 12, 23, 24]),
        },
        "landmarks": pts,
    }
