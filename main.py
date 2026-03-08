import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from enum import Enum, auto

class State(Enum):
    REST = auto()
    INIT = auto()
    PROGRESS = auto()
    RECOVER = auto()

# ── angle helper ──────────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    """Angle at point b, formed by a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

# ── pose correctness ──────────────────────────────────────────────────────────
KNEE_HIP_SHOULDER_TARGET  = (140, 180)   # bridge: hips lifted, ~160°
FOOT_KNEE_HIP_TARGET      = (60,  80)    # ~45°
KNEE_FOOT_SHOULDER_TARGET = (75, 105)    # ~90°

def is_correct_pose(shoulder, hip, knee, ankle):
    a1 = calculate_angle(knee,    hip,   shoulder)  # knee-hip-shoulder
    a2 = calculate_angle(ankle,   knee,  hip)        # foot-knee-hip
    a3 = calculate_angle(knee,    ankle, shoulder)   # knee-foot-shoulder  (shoulder used as far ref)
    ok = (KNEE_HIP_SHOULDER_TARGET[0]  <= a1 <= KNEE_HIP_SHOULDER_TARGET[1]  and
          FOOT_KNEE_HIP_TARGET[0]      <= a2 <= FOOT_KNEE_HIP_TARGET[1]      and
          KNEE_FOOT_SHOULDER_TARGET[0] <= a3 <= KNEE_FOOT_SHOULDER_TARGET[1])
    return ok, a1, a2, a3

# ── color per state ───────────────────────────────────────────────────────────
STATE_COLOR = {
    State.REST:     (255, 255, 255),
    State.INIT:     (255, 255, 255),
    State.PROGRESS: (0,   255,   0),
    State.RECOVER:  (0,     0, 255),
}

# ── mediapipe setup ───────────────────────────────────────────────────────────
model_path = "pose_landmarker_lite.task"
BaseOptions         = python.BaseOptions
PoseLandmarker      = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode   = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)
landmarker = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(2)
print("程式啟動中...按 q 結束")

# ── FSM state ─────────────────────────────────────────────────────────────────
state           = State.REST
state_enter_t   = time.time()          # when we entered the current state
progress_start   = None                # when latest PROGRESS began
progress_elapsed = 0.0                 # seconds in current PROGRESS session
total_progress   = 0.0                 # cumulative seconds across all PROGRESS sessions

# anchor points for the mini-window (set once when entering INIT)
anchor_shoulder = None
anchor_ankle    = None

timestamp = 0

# ── mini-window constants ─────────────────────────────────────────────────────
MINI_W, MINI_H = 220, 260
MINI_X, MINI_Y = 10, 10          # top-left corner on main frame

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    now    = time.time()
    points = None

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        def lm_to_px(lm):
            return [int(lm.x * w), int(lm.y * h)]

        # prefer left side, fall back to right
        for s, hi, k, a in [(11,23,25,27), (12,24,26,28)]:
            if all(landmarks[i].visibility > 0.5 for i in [s, hi, k, a]):
                points = [lm_to_px(landmarks[i]) for i in [s, hi, k, a]]
                break

    # ── FSM transitions ───────────────────────────────────────────────────────
    if points:
        shoulder, hip, knee, ankle = points
        correct, a_khs, a_fkh, a_kfs = is_correct_pose(shoulder, hip, knee, ankle)
    else:
        correct, a_khs, a_fkh, a_kfs = False, 0.0, 0.0, 0.0

    if state == State.REST:
        if correct:
            state         = State.INIT
            state_enter_t = now
            anchor_shoulder = shoulder if points else None
            anchor_ankle    = ankle    if points else None

    elif state == State.INIT:
        if correct:
            if now - state_enter_t >= 2.0:
                state          = State.PROGRESS
                state_enter_t  = now
                progress_start = now
        else:
            state         = State.REST
            state_enter_t = now

    elif state == State.PROGRESS:
        progress_elapsed = now - progress_start
        if not correct:
            total_progress += progress_elapsed
            state         = State.RECOVER
            state_enter_t = now

    elif state == State.RECOVER:
        if correct:
            state          = State.PROGRESS
            state_enter_t  = now
            progress_start = now - progress_elapsed  # resume from where we left off
        elif now - state_enter_t >= 2.0:
            state             = State.REST
            state_enter_t     = now
            progress_start    = None
            progress_elapsed  = 0.0

    color = STATE_COLOR[state]

    # ── draw skeleton on main frame ───────────────────────────────────────────
    if points:
        shoulder, hip, knee, ankle = points

        for p in points:
            cv2.circle(frame, tuple(p), 10, color, cv2.FILLED)

        cv2.line(frame, tuple(shoulder), tuple(hip),   color, 3)
        cv2.line(frame, tuple(hip),      tuple(knee),  color, 3)
        cv2.line(frame, tuple(knee),     tuple(ankle), color, 3)
        cv2.line(frame, tuple(shoulder), tuple(knee),  color, 2)  # diagonal ref

        # angle labels
        for angle_val, ref_pt, dy in [
            (a_khs, hip,   +40),
            (a_fkh, knee,  +40),
            (a_kfs, ankle, +40),
        ]:
            cv2.putText(frame, str(int(angle_val)),
                        (ref_pt[0], ref_pt[1] + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # ── mini sub-window (transparent background) ──────────────────────────────
    mini = frame[MINI_Y:MINI_Y+MINI_H, MINI_X:MINI_X+MINI_W].copy()
    cv2.rectangle(mini, (0,0), (MINI_W-1, MINI_H-1), (80,80,80), 1)

    # ── pose sketch inside mini window ───────────────────────────────────────
    if points:
        shoulder, hip, knee, ankle = points

        # foot (ankle) on bottom-left, shoulder on bottom-right — both fixed
        BOTTOM_Y     = MINI_H - 100
        FIX_FOOT     = np.array([MINI_W // 4,         BOTTOM_Y], dtype=float)
        FIX_SHOULDER = np.array([MINI_W * 3 // 4,     BOTTOM_Y], dtype=float)

        real_foot     = np.array(ankle,    dtype=float)
        real_shoulder = np.array(shoulder, dtype=float)

        # Affine: scale + translate so that real_foot->FIX_FOOT and real_shoulder->FIX_SHOULDER
        real_vec = real_shoulder - real_foot
        fix_vec  = FIX_SHOULDER  - FIX_FOOT
        real_len = np.linalg.norm(real_vec) + 1e-6
        scale    = np.linalg.norm(fix_vec) / real_len
        # rotation angle to align real_vec onto fix_vec
        real_angle = np.arctan2(real_vec[1], real_vec[0])
        fix_angle  = np.arctan2(fix_vec[1],  fix_vec[0])
        dangle     = fix_angle - real_angle
        cos_a, sin_a = np.cos(dangle), np.sin(dangle)

        def map_pt(p):
            rel = (np.array(p, dtype=float) - real_foot) * scale
            rotated = np.array([rel[0]*cos_a - rel[1]*sin_a,
                                 rel[0]*sin_a + rel[1]*cos_a])
            mapped = FIX_FOOT + rotated
            mapped[0] = np.clip(mapped[0], 5, MINI_W - 5)
            mapped[1] = np.clip(mapped[1], 5, MINI_H - 55)
            return (int(mapped[0]), int(mapped[1]))

        ma = (int(FIX_FOOT[0]),     int(FIX_FOOT[1]))      # foot — fixed
        ms = (int(FIX_SHOULDER[0]), int(FIX_SHOULDER[1]))  # shoulder — fixed
        mh = map_pt(hip)
        mk = map_pt(knee)

        # same lines as main window
        cv2.line(mini, ms, mh, color, 2)
        cv2.line(mini, mh, mk, color, 2)
        cv2.line(mini, mk, ma, color, 2)
        cv2.line(mini, ms, mk, color, 1)   # diagonal assist line
        for p in [ms, mh, mk, ma]:
            cv2.circle(mini, p, 5, color, cv2.FILLED)

    # ── state label ───────────────────────────────────────────────────────────
    state_label = state.name
    cv2.putText(mini, f"State: {state_label}",
                (6, MINI_H - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ── current session timer ─────────────────────────────────────────────────
    display_secs = progress_elapsed
    if state == State.PROGRESS and progress_start is not None:
        display_secs = now - progress_start
    mins = int(display_secs) // 60
    secs = int(display_secs) % 60
    cv2.putText(mini, f"Time: {mins:02d}:{secs:02d}",
                (6, MINI_H - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ── total accumulated progress timer ──────────────────────────────────────
    live_total = total_progress
    if state == State.PROGRESS and progress_start is not None:
        live_total += now - progress_start
    t_mins = int(live_total) // 60
    t_secs = int(live_total) % 60
    cv2.putText(mini, f"Total: {t_mins:02d}:{t_secs:02d}",
                (6, MINI_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # ── paste mini window onto main frame ─────────────────────────────────────
    frame[MINI_Y:MINI_Y+MINI_H, MINI_X:MINI_X+MINI_W] = mini

    cv2.imshow("Bridge Exercise", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()