import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r".\outputs\yolo11n_baseline\weights\best.pt"
VIDEO_PATH = r"C:\Users\thavi\Downloads\SAM ANNA.mp4"
OUTPUT_PATH = r".\outputs\videos\hybrid_yolo_klt_tracking_v2.mp4"

CONF = 0.20
IMG_SIZE = 640
YOLO_INTERVAL = 10

# KLT parameters
MAX_CORNERS = 40

feature_params = dict(
    maxCorners=MAX_CORNERS,
    qualityLevel=0.001,
    minDistance=3,
    blockSize=7
)

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)

# ============================================================
# LOAD
# ============================================================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("=====================================")
print("HYBRID YOLO + KLT TRACKER V2")
print("=====================================")
print(f"Video FPS     : {fps:.2f}")
print(f"Resolution    : {width}x{height}")
print(f"Frames        : {total_frames}")
print(f"YOLO interval : {YOLO_INTERVAL}")
print("=====================================")

# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

# ============================================================
# VARIABLES
# ============================================================

paused = True
target_locked = False

target_box = None
prev_gray = None
prev_points = None

trajectory = deque(maxlen=80)

target_conf = 0.0

frame_id = 0

last_center = None
last_time = None

vx = 0.0
vy = 0.0
speed = 0.0

# Prevent the SPACE used to resume from being processed twice
ignore_keys_until = 0

# ============================================================
# MOUSE
# ============================================================

mouse_x = 0
mouse_y = 0


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x = x
    mouse_y = y


cv2.namedWindow("HYBRID YOLO + KLT")
cv2.setMouseCallback("HYBRID YOLO + KLT", mouse_callback)

# ============================================================
# HELPER: FIND YOLO BOX UNDER MOUSE
# ============================================================

def get_detections(frame):
    results = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        device=0,
        verbose=False
    )

    detections = []

    if len(results) == 0:
        return detections

    result = results[0]

    if result.boxes is None:
        return detections

    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())

        x1, y1, x2, y2 = map(int, xyxy)

        detections.append(
            (x1, y1, x2, y2, conf)
        )

    return detections


# ============================================================
# HELPER: CREATE KLT FEATURES
# ============================================================

def create_klt_points(gray, box):

    x1, y1, x2, y2 = box

    # Expand search region
    pad_x = max(10, int((x2 - x1) * 0.8))
    pad_y = max(10, int((y2 - y1) * 1.2))

    sx1 = max(0, x1 - pad_x)
    sy1 = max(0, y1 - pad_y)

    sx2 = min(gray.shape[1], x2 + pad_x)
    sy2 = min(gray.shape[0], y2 + pad_y)

    mask = np.zeros_like(gray)

    mask[sy1:sy2, sx1:sx2] = 255

    points = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        **feature_params
    )

    return points


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("\nVideo finished.")
        break

    frame_id += 1

    display = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ========================================================
    # PAUSED MODE
    # ========================================================

    if paused:

        detections = get_detections(frame)

        # Draw YOLO detections
        for i, (x1, y1, x2, y2, conf) in enumerate(detections):

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                2
            )

            cv2.putText(
                display,
                f"DRONE {i+1} {conf:.2f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2
            )

        cv2.putText(
            display,
            "PAUSED - CLICK DRONE TO LOCK",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "SPACE = resume | Q = quit",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Mouse click
        # ----------------------------------------------------

        if cv2.getWindowProperty(
            "HYBRID YOLO + KLT",
            cv2.WND_PROP_VISIBLE
        ) < 1:

            break

        # Detect left click using callback state indirectly:
        # User clicks while paused and then SPACE to lock.
        #
        # Instead of relying on mouse event state,
        # allow SPACE to lock the currently hovered YOLO box.

    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow("HYBRID YOLO + KLT", display)

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q") or key == 27:
        print("\nUser requested exit.")
        break

    # --------------------------------------------------------
    # SPACE
    # --------------------------------------------------------

    if key == 32:

        # ----------------------------------------------
        # PAUSED -> LOCK
        # ----------------------------------------------

        if paused:

            detections = get_detections(frame)

            selected = None

            # Find detection containing mouse
            for det in detections:

                x1, y1, x2, y2, conf = det

                if (
                    x1 <= mouse_x <= x2
                    and
                    y1 <= mouse_y <= y2
                ):
                    selected = det
                    break

            # If mouse isn't directly inside a box,
            # choose nearest detection center.
            if selected is None and detections:

                best_dist = float("inf")

                for det in detections:

                    x1, y1, x2, y2, conf = det

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    dist = (
                        (cx - mouse_x) ** 2
                        +
                        (cy - mouse_y) ** 2
                    )

                    if dist < best_dist:
                        best_dist = dist
                        selected = det

            if selected is None:

                print("[LOCK FAILED] No YOLO detection found.")

                continue

            x1, y1, x2, y2, conf = selected

            target_box = (x1, y1, x2, y2)
            target_conf = conf

            prev_points = create_klt_points(
                gray,
                target_box
            )

            if prev_points is None:

                print("[LOCK FAILED] No KLT features.")

                continue

            print()
            print("================================")
            print("TARGET LOCKED")
            print(
                f"Box: ({x1}, {y1}, {x2}, {y2})"
            )
            print(
                f"KLT points: {len(prev_points)}"
            )
            print("Press SPACE to resume tracking.")
            print("================================")

            target_locked = True

            # Important:
            # Clear previous trajectory/state
            trajectory.clear()

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            trajectory.append((cx, cy))

            last_center = (cx, cy)
            last_time = time.time()

            paused = False

            # Save current grayscale frame
            prev_gray = gray.copy()

            # Ignore keyboard events for a short moment
            # so the same SPACE press cannot toggle again.
            cv2.waitKey(150)

            print("[RESUMED] Tracking.")

            continue

        # ----------------------------------------------
        # TRACKING -> PAUSE
        # ----------------------------------------------

        else:

            paused = True

            print("[PAUSED] Tracking paused.")

            continue

    # ========================================================
    # TRACKING MODE
    # ========================================================

    if target_locked and not paused:

        # ----------------------------------------------------
        # KLT
        # ----------------------------------------------------

        if (
            prev_points is not None
            and
            len(prev_points) > 0
        ):

            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                prev_points,
                None,
                **lk_params
            )

            if new_points is not None:

                good_new = new_points[
                    status.flatten() == 1
                ]

                good_old = prev_points[
                    status.flatten() == 1
                ]

                if len(good_new) >= 2:

                    dx = np.median(
                        good_new[:, 0]
                        -
                        good_old[:, 0]
                    )

                    dy = np.median(
                        good_new[:, 1]
                        -
                        good_old[:, 1]
                    )

                    x1, y1, x2, y2 = target_box

                    x1 += int(dx)
                    x2 += int(dx)

                    y1 += int(dy)
                    y2 += int(dy)

                    # Clamp to frame
                    x1 = max(0, min(width - 1, x1))
                    y1 = max(0, min(height - 1, y1))
                    x2 = max(1, min(width, x2))
                    y2 = max(1, min(height, y2))

                    target_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    prev_points = good_new.reshape(
                        -1, 1, 2
                    )

                else:

                    prev_points = None

        # ----------------------------------------------------
        # YOLO CORRECTION
        # ----------------------------------------------------

        if frame_id % YOLO_INTERVAL == 0:

            detections = get_detections(frame)

            if detections:

                tx1, ty1, tx2, ty2 = target_box

                tcx = (tx1 + tx2) / 2
                tcy = (ty1 + ty2) / 2

                best = None
                best_dist = float("inf")

                for det in detections:

                    x1, y1, x2, y2, conf = det

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    dist = (
                        (cx - tcx) ** 2
                        +
                        (cy - tcy) ** 2
                    )

                    if dist < best_dist:

                        best_dist = dist
                        best = det

                if best is not None:

                    x1, y1, x2, y2, conf = best

                    target_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    target_conf = conf

                    # Refresh KLT features
                    new_points = create_klt_points(
                        gray,
                        target_box
                    )

                    if new_points is not None:

                        prev_points = new_points

                    print(
                        f"[YOLO CORRECTION] "
                        f"Frame {frame_id} "
                        f"Conf {conf:.2f}"
                    )

        # ----------------------------------------------------
        # TARGET CENTER
        # ----------------------------------------------------

        x1, y1, x2, y2 = target_box

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        trajectory.append((cx, cy))

        # ----------------------------------------------------
        # VELOCITY
        # ----------------------------------------------------

        current_time = time.time()

        if last_center is not None:

            dt = current_time - last_time

            if dt > 0:

                vx = (
                    cx - last_center[0]
                ) / dt

                vy = (
                    cy - last_center[1]
                ) / dt

                speed = np.sqrt(
                    vx * vx +
                    vy * vy
                )

        last_center = (cx, cy)
        last_time = current_time

        # ----------------------------------------------------
        # DRAW TARGET
        # ----------------------------------------------------

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            display,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )

        # ----------------------------------------------------
        # DRAW KLT POINTS
        # ----------------------------------------------------

        if prev_points is not None:

            for point in prev_points:

                px, py = point.ravel()

                cv2.circle(
                    display,
                    (int(px), int(py)),
                    3,
                    (255, 255, 0),
                    -1
                )

        # ----------------------------------------------------
        # TRAJECTORY
        # ----------------------------------------------------

        for i in range(1, len(trajectory)):

            cv2.line(
                display,
                trajectory[i - 1],
                trajectory[i],
                (255, 0, 255),
                2
            )

        # ----------------------------------------------------
        # PREDICTED POSITION
        # ----------------------------------------------------

        prediction_time = 0.2

        pred_x = int(
            cx + vx * prediction_time
        )

        pred_y = int(
            cy + vy * prediction_time
        )

        cv2.circle(
            display,
            (pred_x, pred_y),
            7,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            display,
            "TARGET LOCKED",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        klt_count = (
            0
            if prev_points is None
            else len(prev_points)
        )

        cv2.putText(
            display,
            f"KLT: {klt_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"VX: {vx:.1f} px/s",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"VY: {vy:.1f} px/s",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Speed: {speed:.1f} px/s",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Frame: {frame_id}/{total_frames}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "SPACE = pause | Q = quit",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # ========================================================
    # WRITE FRAME
    # ========================================================

    writer.write(display)

# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()
cv2.destroyAllWindows()

print() 
print("=====================================")
print("TRACKING FINISHED")
print("=====================================")
print(f"Output: {OUTPUT_PATH}")
print(f"Frames processed: {frame_id}")