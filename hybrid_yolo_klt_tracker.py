import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math
import os

# ============================================================
# CONFIGURATION
# ============================================================

MODEL = r".\outputs\yolo11n_baseline\weights\best.pt"
VIDEO = r"C:\Users\thavi\Downloads\SAM ANNA.mp4"

OUTPUT = r".\outputs\videos\hybrid_yolo_klt_tracking.mp4"

CONF = 0.20
IMG_SIZE = 640

# YOLO correction interval.
# 10 means YOLO is used every 10 frames while KLT runs between them.
YOLO_INTERVAL = 10

# Minimum KLT points before we consider KLT weak.
MIN_KLT_POINTS = 2

# Maximum trajectory points to display.
TRAIL_LENGTH = 80

# KLT parameters
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    ),
)

# ============================================================
# LOAD MODEL / VIDEO
# ============================================================

print("[INFO] Loading YOLO model...")

model = YOLO(MODEL)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO}"
    )

fps_input = cap.get(cv2.CAP_PROP_FPS)

if fps_input <= 0:
    fps_input = 30.0

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print()
print("======================================")
print("HYBRID YOLO + KLT TRACKER")
print("======================================")
print(f"Video FPS     : {fps_input:.2f}")
print(f"Resolution    : {width}x{height}")
print(f"Frames        : {total_frames}")
print(f"YOLO interval : {YOLO_INTERVAL}")
print("======================================")
print()

# ============================================================
# OUTPUT VIDEO
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT,
    fourcc,
    fps_input,
    (width, height)
)

# ============================================================
# STATE
# ============================================================

paused = False
locked = False

frame = None
gray = None
previous_gray = None

target_box = None

klt_points = None

detections = []
confidences = []

trajectory = deque(
    maxlen=TRAIL_LENGTH
)

frame_number = 0

last_yolo_frame = -999

target_confidence = 0.0

velocity_x = 0.0
velocity_y = 0.0
velocity = 0.0

lost_count = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def detect(frame):

    result = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        device=0,
        verbose=False,
    )[0]

    boxes = []
    confs = []

    if result.boxes is None:
        return boxes, confs

    xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    for box, score in zip(xyxy, scores):

        x1, y1, x2, y2 = map(
            int,
            box
        )

        boxes.append(
            (x1, y1, x2, y2)
        )

        confs.append(
            float(score)
        )

    return boxes, confs


def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def box_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(
        0,
        ay2 - ay1
    )

    area_b = max(0, bx2 - bx1) * max(
        0,
        by2 - by1
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def expand_box(box, shape, scale=2.0):

    h, w = shape[:2]

    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    nx1 = int(
        max(0, cx - bw / 2)
    )

    ny1 = int(
        max(0, cy - bh / 2)
    )

    nx2 = int(
        min(w - 1, cx + bw / 2)
    )

    ny2 = int(
        min(h - 1, cy + bh / 2)
    )

    return (
        nx1,
        ny1,
        nx2,
        ny2
    )


def get_klt_features(gray_image, box):

    # Search a larger region around the target.
    x1, y1, x2, y2 = expand_box(
        box,
        gray_image.shape,
        scale=2.0
    )

    mask = np.zeros_like(
        gray_image
    )

    cv2.rectangle(
        mask,
        (x1, y1),
        (x2, y2),
        255,
        -1
    )

    points = cv2.goodFeaturesToTrack(
        gray_image,
        mask=mask,
        maxCorners=80,
        qualityLevel=0.001,
        minDistance=2,
        blockSize=3
    )

    return points


def initialize_klt(box):

    global klt_points

    points = get_klt_features(
        gray,
        box
    )

    if points is None:
        klt_points = None
        return 0

    klt_points = points

    return len(points)


def clamp_box(box):

    x1, y1, x2, y2 = box

    x1 = max(
        0,
        min(width - 1, int(x1))
    )

    y1 = max(
        0,
        min(height - 1, int(y1))
    )

    x2 = max(
        0,
        min(width - 1, int(x2))
    )

    y2 = max(
        0,
        min(height - 1, int(y2))
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


def find_best_detection(
    boxes,
    confs,
    current_box
):

    if not boxes:
        return None

    if current_box is None:

        best_index = int(
            np.argmax(confs)
        )

        return (
            boxes[best_index],
            confs[best_index]
        )

    current_center = np.array(
        box_center(current_box)
    )

    best_index = None
    best_score = -999999

    for i, box in enumerate(boxes):

        center = np.array(
            box_center(box)
        )

        distance = np.linalg.norm(
            center - current_center
        )

        iou = box_iou(
            current_box,
            box
        )

        # Distance is normalized approximately
        # relative to the image diagonal.
        diagonal = math.sqrt(
            width ** 2 +
            height ** 2
        )

        normalized_distance = (
            distance / diagonal
        )

        score = (
            3.0 * iou
            + 1.0 * confs[i]
            - 2.0 * normalized_distance
        )

        if score > best_score:

            best_score = score
            best_index = i

    if best_index is None:
        return None

    return (
        boxes[best_index],
        confs[best_index]
    )


# ============================================================
# MOUSE SELECTION
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global locked
    global target_box
    global klt_points
    global target_confidence
    global trajectory

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if not paused:

        print(
            "[INFO] Press SPACE first "
            "to pause."
        )

        return

    for i, box in enumerate(
        detections
    ):

        x1, y1, x2, y2 = box

        if (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        ):

            target_box = box

            target_confidence = (
                confidences[i]
            )

            points_count = (
                initialize_klt(
                    target_box
                )
            )

            # We allow even one point.
            # YOLO will handle recovery.
            if points_count >= 1:

                locked = True

                trajectory.clear()

                cx, cy = box_center(
                    target_box
                )

                trajectory.append(
                    (int(cx), int(cy))
                )

                print()
                print(
                    "================================"
                )
                print(
                    "TARGET LOCKED"
                )
                print(
                    f"Box: {target_box}"
                )
                print(
                    f"KLT points: {points_count}"
                )
                print(
                    "Press SPACE to resume."
                )
                print(
                    "================================"
                )
                print()

            else:

                print(
                    "[LOCK FAILED] "
                    "No KLT points."
                )

            return

    print(
        "[INFO] Click directly "
        "inside an orange YOLO box."
    )


# ============================================================
# WINDOW
# ============================================================

cv2.namedWindow(
    "Hybrid YOLO + KLT",
    cv2.WINDOW_NORMAL
)

cv2.setMouseCallback(
    "Hybrid YOLO + KLT",
    mouse_callback
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    if not paused:

        ret, new_frame = cap.read()

        if not ret:
            break

        frame = new_frame

        frame_number += 1

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # INITIAL YOLO DETECTION
        # ----------------------------------------------------

        if not locked:

            detections, confidences = detect(
                frame
            )

        # ----------------------------------------------------
        # TRACKING
        # ----------------------------------------------------

        else:

            # ================================================
            # KLT
            # ================================================

            if (
                previous_gray is not None
                and
                klt_points is not None
            ):

                new_points, status, error = (
                    cv2.calcOpticalFlowPyrLK(
                        previous_gray,
                        gray,
                        klt_points,
                        None,
                        **lk_params
                    )
                )

                if new_points is not None:

                    status = status.flatten()

                    good_new = (
                        new_points[status == 1]
                    )

                    good_old = (
                        klt_points[status == 1]
                    )

                    if len(good_new) >= 1:

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

                        x1, y1, x2, y2 = (
                            target_box
                        )

                        target_box = clamp_box(
                            (
                                x1 + dx,
                                y1 + dy,
                                x2 + dx,
                                y2 + dy
                            )
                        )

                        klt_points = (
                            good_new
                            .reshape(-1, 1, 2)
                        )

                        lost_count = 0

                    else:

                        lost_count += 1

                else:

                    lost_count += 1

            else:

                lost_count += 1

            # ================================================
            # YOLO CORRECTION
            # ================================================

            should_run_yolo = (
                frame_number -
                last_yolo_frame
                >= YOLO_INTERVAL
            )

            if (
                should_run_yolo
                or
                lost_count >= 2
            ):

                yolo_boxes, yolo_confs = (
                    detect(frame)
                )

                last_yolo_frame = (
                    frame_number
                )

                match = find_best_detection(
                    yolo_boxes,
                    yolo_confs,
                    target_box
                )

                if match is not None:

                    detected_box, detected_conf = (
                        match
                    )

                    # Correct the target box
                    target_box = (
                        int(detected_box[0]),
                        int(detected_box[1]),
                        int(detected_box[2]),
                        int(detected_box[3])
                    )

                    target_confidence = (
                        detected_conf
                    )

                    # Reinitialize KLT
                    # using the corrected YOLO box.
                    points_count = (
                        initialize_klt(
                            target_box
                        )
                    )

                    lost_count = 0

                    print(
                        f"[YOLO CORRECTION] "
                        f"Frame {frame_number} | "
                        f"Conf {detected_conf:.2f} | "
                        f"KLT {points_count}"
                    )

                else:

                    lost_count += 1

                    print(
                        f"[YOLO] Target not found "
                        f"at frame {frame_number}"
                    )

            # ================================================
            # TARGET POSITION
            # ================================================

            cx, cy = box_center(
                target_box
            )

            trajectory.append(
                (int(cx), int(cy))
            )

            # ================================================
            # VELOCITY
            # ================================================

            if len(trajectory) >= 2:

                x_old, y_old = trajectory[-2]
                x_new, y_new = trajectory[-1]

                velocity_x = (
                    x_new - x_old
                ) * fps_input

                velocity_y = (
                    y_new - y_old
                ) * fps_input

                velocity = math.sqrt(
                    velocity_x ** 2
                    +
                    velocity_y ** 2
                )

        previous_gray = gray.copy()

    # ========================================================
    # DRAW
    # ========================================================

    if frame is None:
        continue

    display = frame.copy()

    # --------------------------------------------------------
    # UNLOCKED
    # --------------------------------------------------------

    if not locked:

        for i, (
            box,
            conf
        ) in enumerate(
            zip(
                detections,
                confidences
            )
        ):

            x1, y1, x2, y2 = box

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                2
            )

            cv2.putText(
                display,
                f"{i + 1}: DRONE {conf:.2f}",
                (
                    x1,
                    max(25, y1 - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2
            )

        cv2.putText(
            display,
            "SPACE = PAUSE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if paused:

            cv2.putText(
                display,
                "CLICK DRONE TO LOCK",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

    # --------------------------------------------------------
    # LOCKED
    # --------------------------------------------------------

    else:

        x1, y1, x2, y2 = (
            target_box
        )

        # Target box
        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # KLT points
        if klt_points is not None:

            for p in klt_points.reshape(
                -1,
                2
            ):

                px, py = map(
                    int,
                    p
                )

                cv2.circle(
                    display,
                    (px, py),
                    3,
                    (255, 0, 0),
                    -1
                )

        # Center
        cx, cy = box_center(
            target_box
        )

        cx = int(cx)
        cy = int(cy)

        cv2.circle(
            display,
            (cx, cy),
            6,
            (0, 255, 255),
            -1
        )

        # Trajectory
        if len(trajectory) > 1:

            pts = np.array(
                trajectory,
                dtype=np.int32
            )

            cv2.polylines(
                display,
                [pts],
                False,
                (255, 255, 0),
                2
            )

        # Prediction
        predicted_x = int(
            cx + velocity_x * 0.20
        )

        predicted_y = int(
            cy + velocity_y * 0.20
        )

        cv2.circle(
            display,
            (
                predicted_x,
                predicted_y
            ),
            7,
            (0, 0, 255),
            2
        )

        cv2.line(
            display,
            (cx, cy),
            (
                predicted_x,
                predicted_y
            ),
            (0, 0, 255),
            2
        )

        # Text panel
        cv2.putText(
            display,
            "TARGET LOCKED",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        klt_count = (
            0
            if klt_points is None
            else len(klt_points)
        )

        cv2.putText(
            display,
            f"KLT: {klt_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"CONF: {target_confidence:.2f}",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"POS: {cx}, {cy}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"VX: {velocity_x:.1f} px/s",
            (10, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"VY: {velocity_y:.1f} px/s",
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"SPEED: {velocity:.1f} px/s",
            (10, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"FRAME: {frame_number}",
            (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "SPACE = PAUSE | Q = QUIT",
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        display,
        f"VIDEO FPS: {fps_input:.1f}",
        (
            width - 180,
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # SHOW + SAVE
    # --------------------------------------------------------

    cv2.imshow(
        "Hybrid YOLO + KLT",
        display
    )

    writer.write(
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord(" "):

        paused = not paused

        if paused:

            print(
                "[PAUSED] "
                "Click the drone to lock it."
            )

        else:

            print(
                "[RESUMED] Tracking."
            )

# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()

cv2.destroyAllWindows()

print()
print("======================================")
print("TRACKING COMPLETE")
print("======================================")
print(f"Output: {os.path.abspath(OUTPUT)}")
print(f"Frames processed: {frame_number}")
print("======================================")