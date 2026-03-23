import cv2
import mediapipe as mp
import numpy as np
import os
import time
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── Model download ──────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

# ── Tuning ──────────────────────────────────────────────────────
SMOOTH       = 0.25
DEAD_ZONE    = 6
BRUSH_SIZE   = 7
HOVER_FRAMES = 12   # frames to hold over swatch before selecting

# ── Colors ──────────────────────────────────────────────────────
COLORS = [
    ("Red",    (0,   0,   255)),
    ("Orange", (0,   128, 255)),
    ("Yellow", (0,   215, 255)),
    ("Green",  (0,   200,  80)),
    ("Blue",   (220, 100,   0)),
    ("Purple", (180,   0, 180)),
    ("White",  (255, 255, 255)),
]

PAL_RIGHT_MARGIN = 20
PAL_SWATCH_H     = 55
PAL_SWATCH_W     = 50
PAL_TOP          = 60

# ── Callback ────────────────────────────────────────────────────
latest_result = None
def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# ── MediaPipe ───────────────────────────────────────────────────
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
    result_callback=on_result,
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_skeleton(frame, lms, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 100), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 80, 0), -1)
    return pts

def count_extended(lms):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return sum(lms[t].y < lms[p].y for t, p in zip(tips, pips))

def is_peace(lms):
    """Index + middle up, ring + pinky down."""
    f = [lms[t].y < lms[p].y for t, p in zip([8,12,16,20],[6,10,14,18])]
    return f[0] and f[1] and not f[2] and not f[3]

def build_palette_rects(w):
    rects = []
    for i in range(len(COLORS)):
        x1 = w - PAL_RIGHT_MARGIN - PAL_SWATCH_W
        y1 = PAL_TOP + i * PAL_SWATCH_H
        x2 = w - PAL_RIGHT_MARGIN
        y2 = y1 + PAL_SWATCH_H - 4
        rects.append((x1, y1, x2, y2))
    return rects

def draw_palette(frame, rects, sel_idx, hover_idx=-1, hover_progress=0.0):
    for i, (x1, y1, x2, y2) in enumerate(rects):
        name, color = COLORS[i]
        # shadow
        cv2.rectangle(frame, (x1+2, y1+2), (x2+2, y2+2), (0,0,0), -1)
        # swatch
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        # selected ring
        if i == sel_idx:
            cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255,255,255), 2)
        # label
        cv2.putText(frame, name, (x1 - 58, (y1+y2)//2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        # hover highlight + progress arc
        if i == hover_idx:
            cv2.rectangle(frame, (x1-4, y1-4), (x2+4, y2+4), (255,255,0), 2)
            # progress bar at bottom of swatch
            bar_w = int((x2 - x1) * hover_progress)
            cv2.rectangle(frame, (x1, y2+2), (x1+bar_w, y2+6), (255,255,0), -1)

# ── Main ────────────────────────────────────────────────────────
cap    = cv2.VideoCapture(0)
canvas = None

smooth_x: float = None
smooth_y: float = None
prev_pt  = None

sel_color_idx  = 0
BUFFER_SIZE    = 6
gesture_buffer = []

hover_idx    = -1   # which swatch is being hovered
hover_count  = 0    # frames held on that swatch

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_img, timestamp)

        cur_color = COLORS[sel_color_idx][1]
        gesture   = "none"
        pal_rects = build_palette_rects(w)

        if latest_result and latest_result.hand_landmarks:
            lms = latest_result.hand_landmarks[0]
            pts = draw_skeleton(frame, lms, w, h)
            raw_x, raw_y = pts[8]   # index tip

            # smooth index tip
            if smooth_x is None:
                smooth_x, smooth_y = float(raw_x), float(raw_y)
            else:
                smooth_x = SMOOTH * raw_x + (1 - SMOOTH) * smooth_x
                smooth_y = SMOOTH * raw_y + (1 - SMOOTH) * smooth_y
            ix, iy = int(smooth_x), int(smooth_y)

            if is_peace(lms):
                gesture = "peace"

                # use midpoint of index + middle tip as pointer
                mx = int((pts[8][0] + pts[12][0]) / 2)
                my = int((pts[8][1] + pts[12][1]) / 2)

                # draw pointer dot between the two fingers
                cv2.circle(frame, (mx, my), 7, (255, 255, 0), -1)
                cv2.line(frame, pts[8], pts[12], (255,255,0), 2)

                # check which swatch is hit
                hit = -1
                for i, (x1, y1, x2, y2) in enumerate(pal_rects):
                    if x1 <= mx <= x2 and y1 <= my <= y2:
                        hit = i
                        break

                if hit == hover_idx and hit >= 0:
                    hover_count += 1
                    if hover_count >= HOVER_FRAMES:
                        sel_color_idx = hit
                        cur_color     = COLORS[sel_color_idx][1]
                        hover_count   = 0   # reset so it can re-select
                else:
                    hover_idx   = hit
                    hover_count = 0

            else:
                hover_idx   = -1
                hover_count = 0
                n = count_extended(lms)
                if n >= 3:
                    gesture = "erase"
                elif n <= 1:
                    gesture = "draw"

        else:
            smooth_x = smooth_y = None
            prev_pt  = None
            hover_idx   = -1
            hover_count = 0

        # majority vote (only for draw/erase)
        if gesture in ("draw", "erase", "none"):
            gesture_buffer.append(gesture)
            if len(gesture_buffer) > BUFFER_SIZE:
                gesture_buffer.pop(0)
            draw_votes  = gesture_buffer.count("draw")
            erase_votes = gesture_buffer.count("erase")
            if erase_votes > BUFFER_SIZE // 2:
                confirmed = "erase"
            elif draw_votes > BUFFER_SIZE // 2:
                confirmed = "draw"
            else:
                confirmed = "none"
        else:
            confirmed = "peace"
            gesture_buffer.clear()
            prev_pt = None

        # Act
        if confirmed == "draw" and smooth_x is not None:
            if prev_pt is None:
                prev_pt = (ix, iy)
            else:
                dx   = ix - prev_pt[0]
                dy   = iy - prev_pt[1]
                if (dx*dx + dy*dy) ** 0.5 >= DEAD_ZONE:
                    cv2.line(canvas, prev_pt, (ix, iy), cur_color, BRUSH_SIZE)
                    prev_pt = (ix, iy)
            cv2.circle(frame, (ix, iy), BRUSH_SIZE + 2, cur_color, 2)
            cv2.circle(frame, (ix, iy), 2, (255,255,255), -1)

        elif confirmed == "erase":
            canvas  = np.zeros((h, w, 3), dtype=np.uint8)
            prev_pt = None

        else:
            prev_pt = None

        # Compose
        output = cv2.addWeighted(frame, 0.6, canvas, 0.9, 0)

        # Palette on output
        hover_progress = hover_count / HOVER_FRAMES if hover_idx >= 0 else 0.0
        draw_palette(output, pal_rects, sel_color_idx, hover_idx, hover_progress)

        # HUD
        label_map  = {"draw":"DRAW","erase":"ERASE","peace":"PICK COLOR","none":"IDLE"}
        color_map  = {"draw":cur_color,"erase":(0,200,255),"peace":(255,255,0),"none":(160,160,160)}
        label      = label_map.get(confirmed, "IDLE")
        lc         = color_map.get(confirmed, (160,160,160))
        cv2.rectangle(output, (0,0), (w, 48), (20,20,20), -1)
        cv2.circle(output, (22, 24), 12, cur_color, -1)
        cv2.putText(output, label, (44, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, lc, 2)
        cv2.putText(output, "Index=Draw | Palm=Erase | 2 fingers on swatch=Color | ESC=Quit",
                    (190, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130,130,130), 1)

        cv2.imshow("Virtual Drawing", output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
