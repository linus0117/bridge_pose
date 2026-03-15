import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from enum import Enum, auto
import tkinter as tk
from tkinter import font as tkfont


# ── States ────────────────────────────────────────────────────────────────────
class State(Enum):
    INIT   = auto()   # waiting for first bridge pose
    STAGE1 = auto()   # in bridge pose  (hips lifted, angle 140–180°)
    STAGE2 = auto()   # resting between reps


# ── Angle helper ───────────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    """Returns the angle (degrees) at vertex b formed by a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


# ── Bridge-pose detection ──────────────────────────────────────────────────────
#   Primary criterion: knee-hip-shoulder angle must be 140–180°
#   (hips fully lifted off the ground)
BRIDGE_MIN = 140
BRIDGE_MAX = 180


def detect_bridge(shoulder, hip, knee, ankle):
    """
    Returns (in_bridge, a_khs, a_fkh, a_kfs).
    in_bridge is True when the knee-hip-shoulder angle is in [140, 180].
    """
    a_khs = calculate_angle(knee,  hip,   shoulder)   # main bridge angle
    a_fkh = calculate_angle(ankle, knee,  hip)         # leg bend reference
    a_kfs = calculate_angle(knee,  ankle, shoulder)    # foot-reference
    in_bridge = BRIDGE_MIN <= a_khs <= BRIDGE_MAX
    return in_bridge, a_khs, a_fkh, a_kfs


# ── Colour per state ──────────────────────────────────────────────────────────
STATE_COLOR = {
    State.INIT:   (200, 200, 200),   # grey
    State.STAGE1: (0,   255,   0),   # green  – in pose
    State.STAGE2: (0,   165, 255),   # orange – resting
}


# ── Duration picker dialog ────────────────────────────────────────────────────
def ask_duration() -> int:
    """
    Shows a modal dialog with 4 buttons: 1 / 3 / 5 / 10 minutes.
    Returns the chosen duration in seconds.
    Closes the app if the user dismisses the window without choosing.
    """
    chosen = [None]   # mutable container so the inner callbacks can write to it

    root = tk.Tk()
    root.title("橋式訓練 / Bridge Exercise")
    root.resizable(False, False)

    # ── Center the window on screen ───────────────────────────────────────────
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()

    # ── Styles ────────────────────────────────────────────────────────────────
    BG       = "#1e1e2e"
    ACCENT   = "#89b4fa"    # pastel blue
    BTN_BG   = "#313244"
    BTN_FG   = "#cdd6f4"
    BTN_HOV  = "#45475a"
    TITLE_FG = "#cba6f7"    # mauve

    root.configure(bg=BG)

    title_font  = tkfont.Font(family="Helvetica", size=16, weight="bold")
    sub_font    = tkfont.Font(family="Helvetica", size=11)
    btn_font    = tkfont.Font(family="Helvetica", size=18, weight="bold")
    label_font  = tkfont.Font(family="Helvetica", size=9)

    # ── Title ─────────────────────────────────────────────────────────────────
    tk.Label(root, text="🧘 橋式訓練計時器", font=title_font,
             bg=BG, fg=TITLE_FG, pady=14).pack()
    tk.Label(root, text="請選擇訓練時間 / Choose session duration",
             font=sub_font, bg=BG, fg=BTN_FG).pack(pady=(0, 16))

    btn_frame = tk.Frame(root, bg=BG)
    btn_frame.pack(padx=30, pady=(0, 8))

    OPTIONS = [
        ("1 min",  1 * 60),
        ("3 min",  3 * 60),
        ("5 min",  5 * 60),
        ("10 min", 10 * 60),
    ]

    def make_handler(secs):
        def handler():
            chosen[0] = secs
            root.destroy()
        return handler

    def on_enter(btn):
        btn.configure(bg=BTN_HOV)

    def on_leave(btn):
        btn.configure(bg=BTN_BG)

    for label, secs in OPTIONS:
        b = tk.Button(
            btn_frame,
            text=label,
            font=btn_font,
            width=6,
            bg=BTN_BG,
            fg=ACCENT,
            activebackground=BTN_HOV,
            activeforeground=ACCENT,
            relief="flat",
            bd=0,
            padx=10,
            pady=12,
            cursor="hand2",
            command=make_handler(secs),
        )
        b.pack(side="left", padx=8, pady=4)
        b.bind("<Enter>", lambda e, btn=b: on_enter(btn))
        b.bind("<Leave>", lambda e, btn=b: on_leave(btn))

    tk.Label(root, text="計時從第一個橋式動作開始 / Timer starts on your first rep",
             font=label_font, bg=BG, fg="#6c7086", pady=10).pack()

    # ── Size & center ─────────────────────────────────────────────────────────
    root.update_idletasks()
    ww, wh = root.winfo_width(), root.winfo_height()
    root.geometry(f"+{(sw - ww) // 2}+{(sh - wh) // 2}")

    root.protocol("WM_DELETE_WINDOW", root.destroy)   # X button just closes
    root.mainloop()

    if chosen[0] is None:
        import sys
        sys.exit(0)   # user closed without picking — exit cleanly

    return chosen[0]


# ── Result dialog ─────────────────────────────────────────────────────────────
def show_result_dialog(count: int, total_secs: int):
    mins = total_secs // 60

    root = tk.Tk()
    root.title("訓練結果 / Result")
    root.resizable(False, False)

    # ── Shared palette (same as picker) ───────────────────────────────────────
    BG       = "#1e1e2e"
    ACCENT   = "#89b4fa"   # pastel blue
    BTN_BG   = "#313244"
    BTN_HOV  = "#45475a"
    TITLE_FG = "#cba6f7"   # mauve
    TEXT_FG  = "#cdd6f4"
    GREEN_FG = "#a6e3a1"   # green for the big count
    MUTED    = "#6c7086"

    root.configure(bg=BG)

    title_font  = tkfont.Font(family="Helvetica", size=16, weight="bold")
    sub_font    = tkfont.Font(family="Helvetica", size=11)
    count_font  = tkfont.Font(family="Helvetica", size=52, weight="bold")
    label_font  = tkfont.Font(family="Helvetica", size=10)
    btn_font    = tkfont.Font(family="Helvetica", size=14, weight="bold")

    # ── Title ─────────────────────────────────────────────────────────────────
    tk.Label(root, text="🏆 訓練完成！", font=title_font,
             bg=BG, fg=TITLE_FG, pady=16).pack()

    # ── Big rep counter ────────────────────────────────────────────────────────
    tk.Label(root, text=str(count), font=count_font,
             bg=BG, fg=GREEN_FG).pack(pady=(0, 4))

    tk.Label(root, text="次橋式 / bridge reps", font=sub_font,
             bg=BG, fg=TEXT_FG).pack()

    # ── Divider ───────────────────────────────────────────────────────────────
    tk.Frame(root, bg=BTN_BG, height=1).pack(fill="x", padx=30, pady=16)

    # ── Duration line ─────────────────────────────────────────────────────────
    tk.Label(root,
             text=f"訓練時間 {mins} 分鐘  /  Session: {mins} minute(s)",
             font=label_font, bg=BG, fg=MUTED).pack(pady=(0, 20))

    # ── Close button ──────────────────────────────────────────────────────────
    def on_enter(e): close_btn.configure(bg=BTN_HOV)
    def on_leave(e): close_btn.configure(bg=BTN_BG)

    close_btn = tk.Button(
        root,
        text="關閉  /  Close",
        font=btn_font,
        bg=BTN_BG,
        fg=ACCENT,
        activebackground=BTN_HOV,
        activeforeground=ACCENT,
        relief="flat",
        bd=0,
        padx=24,
        pady=10,
        cursor="hand2",
        command=root.destroy,
    )
    close_btn.pack(pady=(0, 20))
    close_btn.bind("<Enter>", on_enter)
    close_btn.bind("<Leave>", on_leave)

    # ── Center on screen ──────────────────────────────────────────────────────
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    ww, wh = root.winfo_width(), root.winfo_height()
    root.geometry(f"+{(sw - ww) // 2}+{(sh - wh) // 2}")

    root.mainloop()


# ── MediaPipe setup ────────────────────────────────────────────────────────────
MODEL_PATH = "pose_landmarker_lite.task"
BaseOptions           = python.BaseOptions
PoseLandmarker        = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode     = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
)
landmarker = PoseLandmarker.create_from_options(options)

# ── Ask user for session duration BEFORE opening camera ───────────────────────
COUNTDOWN_SEC = ask_duration()   # returns 60 / 180 / 300 / 600

cap = cv2.VideoCapture(0)
print(f"程式啟動中… 訓練時間 {COUNTDOWN_SEC // 60} 分鐘，按 q 手動結束")

# ── FSM / counter / timer state ────────────────────────────────────────────────
state            = State.INIT
bridge_count     = 0            # how many times the user entered STAGE1
timer_started    = False        # True after first STAGE1 entry
timer_start_time = None         # wall-clock time when timer began
finished         = False        # True when countdown hits 0

timestamp = 0   # MediaPipe video timestamp (integer, increments each frame)

# ── Mini sub-window geometry ───────────────────────────────────────────────────
MINI_W, MINI_H = 230, 290
MINI_X, MINI_Y = 10,  10


# ═════════════════════════════════════════════════════════════════════════════
# Main loop
# ═════════════════════════════════════════════════════════════════════════════
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
    points = None   # [shoulder, hip, knee, ankle] in pixel coords

    # ── Landmark extraction ────────────────────────────────────────────────────
    if result.pose_landmarks:
        lms = result.pose_landmarks[0]

        def lm_to_px(lm):
            return [int(lm.x * w), int(lm.y * h)]

        # Try left side first (indices 11,23,25,31), then right (12,24,26,32)
        for s_i, h_i, k_i, a_i in [(11, 23, 25, 31), (12, 24, 26, 32)]:
            if all(lms[i].visibility > 0.5 for i in [s_i, h_i, k_i, a_i]):
                points = [lm_to_px(lms[i]) for i in [s_i, h_i, k_i, a_i]]
                break

    # ── Pose evaluation ────────────────────────────────────────────────────────
    if points:
        shoulder, hip, knee, ankle = points
        in_bridge, a_khs, a_fkh, a_kfs = detect_bridge(shoulder, hip, knee, ankle)
    else:
        in_bridge, a_khs, a_fkh, a_kfs = False, 0.0, 0.0, 0.0

    # ── Countdown check ────────────────────────────────────────────────────────
    if timer_started and not finished:
        remaining = COUNTDOWN_SEC - (now - timer_start_time)
        if remaining <= 0:
            finished = True

    # ── FSM transitions ────────────────────────────────────────────────────────
    if not finished:
        if state == State.INIT:
            if in_bridge:
                # First rep detected
                state        = State.STAGE1
                bridge_count += 1
                # Start the 1-minute countdown
                if not timer_started:
                    timer_started    = True
                    timer_start_time = now

        elif state == State.STAGE1:
            # User lowered hips → rest phase
            if not in_bridge:
                state = State.STAGE2

        elif state == State.STAGE2:
            # User lifted hips again → new rep
            if in_bridge:
                state        = State.STAGE1
                bridge_count += 1

    color = STATE_COLOR[state]

    # ── Draw skeleton on main frame ────────────────────────────────────────────
    if points:
        shoulder, hip, knee, ankle = points

        for p in points:
            cv2.circle(frame, tuple(p), 10, color, cv2.FILLED)

        cv2.line(frame, tuple(shoulder), tuple(hip),   color, 3)
        cv2.line(frame, tuple(hip),      tuple(knee),  color, 3)
        cv2.line(frame, tuple(knee),     tuple(ankle), color, 3)
        cv2.line(frame, tuple(shoulder), tuple(knee),  color, 2)   # diagonal ref

        # ── Auxiliary line: ankle → knee (dashed, with endpoint dots) ─────────
        AUX_COLOR  = (255, 210, 60)   # golden yellow — distinct from state color
        ARC_RADIUS = 55               # px radius of the angle arc
        DASH_LEN   = 12               # pixels per dash segment
        GAP_LEN    = 7                # pixels per gap

        # ── Auxiliary dashed line: ankle → shoulder ────────────────────────────
        as_vec  = np.array(shoulder, dtype=float) - np.array(ankle, dtype=float)
        as_len  = np.linalg.norm(as_vec)
        as_unit = as_vec / (as_len + 1e-6)

        # Draw dashes along ankle → shoulder
        drawn = 0.0
        draw  = True   # alternate dash / gap
        while drawn < as_len:
            seg   = DASH_LEN if draw else GAP_LEN
            end   = min(drawn + seg, as_len)
            if draw:
                p1 = (np.array(ankle, dtype=float) + as_unit * drawn).astype(int)
                p2 = (np.array(ankle, dtype=float) + as_unit * end).astype(int)
                cv2.line(frame, tuple(p1), tuple(p2), AUX_COLOR, 2)
            drawn += seg
            draw   = not draw

        # Endpoint dots
        cv2.circle(frame, tuple(ankle),    8, AUX_COLOR, cv2.FILLED)
        cv2.circle(frame, tuple(shoulder), 8, AUX_COLOR, cv2.FILLED)

        # ── Arc at ankle between ankle→knee and ankle→shoulder ─────────────────
        ak_vec = np.array(knee,     dtype=float) - np.array(ankle, dtype=float)
        # as_vec already computed above for the dashed line

        # Angles in OpenCV convention: clockwise from +x, y-axis points DOWN
        ang_knee     = float(np.degrees(np.arctan2(ak_vec[1], ak_vec[0])))
        ang_shoulder = float(np.degrees(np.arctan2(as_vec[1], as_vec[0])))

        # Always sweep the shorter arc
        diff = (ang_shoulder - ang_knee) % 360
        if diff <= 180:
            start_ang, end_ang = ang_knee, ang_knee + diff
        else:
            start_ang, end_ang = ang_shoulder, ang_shoulder + (360 - diff)

        cv2.ellipse(frame,
                    tuple(ankle),
                    (ARC_RADIUS, ARC_RADIUS),
                    0,                    # ellipse rotation (0 = circle)
                    start_ang, end_ang,
                    AUX_COLOR, 2)

        # Small tick marks at both arc endpoints for clarity
        for ang_deg in [start_ang, end_ang]:
            ang_rad  = np.radians(ang_deg)
            inner    = np.array(ankle, dtype=float) + (ARC_RADIUS - 6) * np.array([np.cos(ang_rad), np.sin(ang_rad)])
            outer    = np.array(ankle, dtype=float) + (ARC_RADIUS + 6) * np.array([np.cos(ang_rad), np.sin(ang_rad)])
            cv2.line(frame, tuple(inner.astype(int)), tuple(outer.astype(int)), AUX_COLOR, 2)

        # Angle annotation
        for angle_val, ref_pt, dy in [
            (a_khs, hip,   +45),
            (a_fkh, knee,  +45),
            (a_kfs, ankle, +45),
        ]:
            cv2.putText(frame, f"{int(angle_val)}",
                        (ref_pt[0], ref_pt[1] + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    # ── Mini sub-window ────────────────────────────────────────────────────────
    mini = frame[MINI_Y:MINI_Y + MINI_H, MINI_X:MINI_X + MINI_W].copy()
    cv2.rectangle(mini, (0, 0), (MINI_W - 1, MINI_H - 1), (80, 80, 80), 1)

    # ── TOP SECTION: Stage + Count ─────────────────────────────────────────────
    cv2.putText(mini, f"Stage: {state.name}",
                (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    cv2.putText(mini, f"Count: {bridge_count}",
                (6, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)   # cyan, bold

    # ── MIDDLE SECTION: Pose sketch ────────────────────────────────────────────
    if points:
        shoulder, hip, knee, ankle = points

        SKETCH_BOTTOM_Y  = MINI_H - 40   # bottom of sketch area
        SKETCH_TOP_CLIP  = 60            # don't draw above this y in mini

        FIX_FOOT     = np.array([MINI_W // 4,     SKETCH_BOTTOM_Y], dtype=float)
        FIX_SHOULDER = np.array([MINI_W * 3 // 4, SKETCH_BOTTOM_Y], dtype=float)

        real_foot     = np.array(ankle,    dtype=float)
        real_shoulder = np.array(shoulder, dtype=float)

        real_vec = real_shoulder - real_foot
        fix_vec  = FIX_SHOULDER  - FIX_FOOT
        scale    = np.linalg.norm(fix_vec) / (np.linalg.norm(real_vec) + 1e-6)

        real_ang = np.arctan2(real_vec[1], real_vec[0])
        fix_ang  = np.arctan2(fix_vec[1],  fix_vec[0])
        dangle   = fix_ang - real_ang
        cos_a, sin_a = np.cos(dangle), np.sin(dangle)

        def map_pt(p):
            rel = (np.array(p, dtype=float) - real_foot) * scale
            rot = np.array([rel[0] * cos_a - rel[1] * sin_a,
                            rel[0] * sin_a + rel[1] * cos_a])
            mapped = FIX_FOOT + rot
            mapped[0] = np.clip(mapped[0], 5,              MINI_W - 5)
            mapped[1] = np.clip(mapped[1], SKETCH_TOP_CLIP, MINI_H - 40)
            return (int(mapped[0]), int(mapped[1]))

        ma = (int(FIX_FOOT[0]),     int(FIX_FOOT[1]))      # ankle – fixed
        ms = (int(FIX_SHOULDER[0]), int(FIX_SHOULDER[1]))  # shoulder – fixed
        mh = map_pt(hip)
        mk = map_pt(knee)

        cv2.line(mini, ms, mh, color, 2)
        cv2.line(mini, mh, mk, color, 2)
        cv2.line(mini, mk, ma, color, 2)
        cv2.line(mini, ms, mk, color, 1)   # diagonal assist
        for p in [ms, mh, mk, ma]:
            cv2.circle(mini, p, 5, color, cv2.FILLED)

        # ── Mini aux: dashed line between the two fixed bottom points ─────────
        # ma = ankle (bottom-left), ms = shoulder (bottom-right) — both fixed
        MINI_AUX   = (255, 210, 60)
        MINI_DASH  = 6
        MINI_GAP   = 4
        MINI_ARC_R = 22

        # Dashed horizontal line: ankle → shoulder
        total_dx = ms[0] - ma[0]   # purely horizontal since both share SKETCH_BOTTOM_Y
        steps    = int(total_dx / (MINI_DASH + MINI_GAP)) + 1
        for i in range(steps):
            x1 = ma[0] + i * (MINI_DASH + MINI_GAP)
            x2 = min(x1 + MINI_DASH, ms[0])
            if x1 < ms[0]:
                cv2.line(mini, (x1, ma[1]), (x2, ma[1]), MINI_AUX, 1)

        # Endpoint dots on the two bottom points
        cv2.circle(mini, ma, 4, MINI_AUX, cv2.FILLED)
        cv2.circle(mini, ms, 4, MINI_AUX, cv2.FILLED)

        # Arc at ankle (ma) sweeping from ankle→shoulder (0°) to ankle→knee
        m_ak_vec    = np.array(mk, dtype=float) - np.array(ma, dtype=float)
        ang_knee_m  = float(np.degrees(np.arctan2(m_ak_vec[1], m_ak_vec[0])))
        ang_shldr_m = 0.0   # ms is directly to the right of ma (same Y, larger X)

        diff_m = (ang_shldr_m - ang_knee_m) % 360
        if diff_m <= 180:
            s_ang, e_ang = ang_knee_m, ang_knee_m + diff_m
        else:
            s_ang, e_ang = ang_shldr_m, ang_shldr_m + (360 - diff_m)

        cv2.ellipse(mini, ma, (MINI_ARC_R, MINI_ARC_R), 0,
                    s_ang, e_ang, MINI_AUX, 1)

        for ang_deg in [s_ang, e_ang]:
            rad   = np.radians(ang_deg)
            dirv  = np.array([np.cos(rad), np.sin(rad)])
            inner = (np.array(ma, dtype=float) + (MINI_ARC_R - 3) * dirv).astype(int)
            outer = (np.array(ma, dtype=float) + (MINI_ARC_R + 3) * dirv).astype(int)
            cv2.line(mini, tuple(inner), tuple(outer), MINI_AUX, 1)

    # ── BOTTOM SECTION: Countdown timer ───────────────────────────────────────
    if timer_started:
        elapsed   = now - timer_start_time
        remaining = max(0.0, COUNTDOWN_SEC - elapsed)
        r_mins = int(remaining) // 60
        r_secs = int(remaining) % 60
        # Flash red in last 10 seconds
        timer_color = (0, 0, 255) if remaining <= 10 else (255, 255, 255)
        cv2.putText(mini, f"Time: {r_mins:02d}:{r_secs:02d}",
                    (6, MINI_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, timer_color, 1)
    else:
        cv2.putText(mini, "Time: --:-- (do pose to start)",
                    (6, MINI_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1)

    # ── Paste mini window onto main frame ──────────────────────────────────────
    frame[MINI_Y:MINI_Y + MINI_H, MINI_X:MINI_X + MINI_W] = mini

    cv2.imshow("Bridge Exercise", frame)

    # ── Exit conditions ────────────────────────────────────────────────────────
    if finished:
        cv2.destroyAllWindows()
        cap.release()
        show_result_dialog(bridge_count, COUNTDOWN_SEC)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup (if user pressed q before timer expired) ─────────────────────────
cap.release()
cv2.destroyAllWindows()