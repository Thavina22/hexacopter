import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO

# ============================================================
# HYBRID YOLO + KLT TRACKER V3.1
#
# Controls:
#   LEFT CLICK = select / lock detected drone
#   SPACE      = pause / resume tracking
#   R          = reset / unlock target
#   Q / ESC    = quit
#
# Behavior:
#   1. Video opens on the first frame.
#   2. YOLO detects available drones.
#   3. Video remains FROZEN until you select a drone.
#   4. Click a drone to lock it.
#   5. Press SPACE to start tracking.
#   6. KLT tracks frame-by-frame.
#   7. YOLO corrects the target every N frames.
# ============================================================


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r".\outputs\yolo11n_baseline\weights\best.pt"

VIDEO_PATH = r"C:\Users\thavi\Downloads\SAM ANNA.mp4"

OUTPUT_PATH = (
    r".\outputs\videos"
    r"\hybrid_yolo_klt_tracking_v3_1.mp4"
)


# ============================================================
# SETTINGS
# ============================================================

CONF = 0.20

IMG_SIZE = 640

YOLO_INTERVAL = 10

MAX_CORNERS = 40


# ============================================================
# KLT FEATURE SETTINGS
# ============================================================

feature_params = dict(
    maxCorners=MAX_CORNERS,
    qualityLevel=0.001,
    minDistance=3,
    blockSize=7,
)


# ============================================================
# KLT OPTICAL FLOW SETTINGS
# ============================================================

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    ),
)


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded.")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


# ============================================================
# PRINT INFORMATION
# ============================================================

print()
print("=====================================")
print("HYBRID YOLO + KLT TRACKER V3.1")
print("=====================================")
print(f"Video FPS     : {fps:.2f}")
print(f"Resolution    : {width}x{height}")
print(f"Frames        : {total_frames}")
print(f"YOLO interval : {YOLO_INTERVAL}")
print("=====================================")
print("LEFT CLICK = lock drone")
print("SPACE      = pause/resume")
print("R          = reset/unlock")
print("Q / ESC    = quit")
print("=====================================")
print()


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height),
)

if not writer.isOpened():
    raise RuntimeError(
        f"Could not create output video:\n{OUTPUT_PATH}"
    )


# ============================================================
# STATE
# ============================================================

paused = True

target_locked = False

target_box = None

target_conf = 0.0

prev_gray = None

prev_points = None

trajectory = deque(maxlen=80)

frame_id = 0

last_center = None

last_time = None

vx = 0.0

vy = 0.0

speed = 0.0

mouse_x = 0

mouse_y = 0

mouse_clicked = False

detections = []

current_frame = None

current_gray = None

quit_requested = False


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global mouse_x
    global mouse_y
    global mouse_clicked

    mouse_x = x
    mouse_y = y

    if event == cv2.EVENT_LBUTTONDOWN:

        mouse_clicked = True


# ============================================================
# CREATE WINDOW
# ============================================================

WINDOW_NAME = "HYBRID YOLO + KLT V3.1"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    width,
    height
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# ============================================================
# YOLO DETECTION
# ============================================================

def get_detections(frame):

    results = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        device=0,
        verbose=False,
    )

    detections = []

    if not results:
        return detections

    result = results[0]

    if result.boxes is None:
        return detections

    for box in result.boxes:

        xyxy = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        conf = float(
            box.conf[0]
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = map(
            int,
            xyxy
        )

        detections.append(
            (
                x1,
                y1,
                x2,
                y2,
                conf
            )
        )

    return detections


# ============================================================
# CREATE KLT FEATURES
# ============================================================

def create_klt_points(gray, box):

    x1, y1, x2, y2 = box

    box_width = max(
        1,
        x2 - x1
    )

    box_height = max(
        1,
        y2 - y1
    )

    # Expand search area around the detected drone.
    pad_x = max(
        10,
        int(box_width * 0.8)
    )

    pad_y = max(
        10,
        int(box_height * 1.2)
    )

    sx1 = max(
        0,
        x1 - pad_x
    )

    sy1 = max(
        0,
        y1 - pad_y
    )

    sx2 = min(
        gray.shape[1],
        x2 + pad_x
    )

    sy2 = min(
        gray.shape[0],
        y2 + pad_y
    )

    mask = np.zeros_like(gray)

    mask[
        sy1:sy2,
        sx1:sx2
    ] = 255

    points = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        **feature_params
    )

    return points


# ============================================================
# SELECT DETECTION UNDER MOUSE
# ============================================================

def select_detection(
    detections,
    mx,
    my
):

    # --------------------------------------------------------
    # First: exact click inside detection
    # --------------------------------------------------------

    inside = []

    for det in detections:

        x1, y1, x2, y2, conf = det

        if (
            x1 <= mx <= x2
            and
            y1 <= my <= y2
        ):

            inside.append(det)

    if inside:

        return max(
            inside,
            key=lambda d: d[4]
        )


    # --------------------------------------------------------
    # Second: small tolerance around detection
    # --------------------------------------------------------

    tolerance = 30

    best = None

    best_dist = float("inf")

    for det in detections:

        x1, y1, x2, y2, conf = det

        dx = max(
            x1 - mx,
            0,
            mx - x2
        )

        dy = max(
            y1 - my,
            0,
            my - y2
        )

        dist = np.sqrt(
            dx * dx +
            dy * dy
        )

        if (
            dist <= tolerance
            and
            dist < best_dist
        ):

            best_dist = dist

            best = det

    return best


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    display,
    detections
):

    for i, (
        x1,
        y1,
        x2,
        y2,
        conf
    ) in enumerate(detections):

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 165, 255),
            2
        )

        cv2.putText(
            display,
            f"DRONE {i + 1}  {conf:.2f}",
            (
                x1,
                max(
                    20,
                    y1 - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            2
        )


# ============================================================
# INITIAL FRAME
#
# Read ONLY ONE frame.
# This is the important fix.
# ============================================================

ret, current_frame = cap.read()

if not ret:

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    raise RuntimeError(
        "Could not read first video frame."
    )


frame_id = 1


current_gray = cv2.cvtColor(
    current_frame,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# INITIAL YOLO DETECTION
# ============================================================

print("Running YOLO on first frame...")

detections = get_detections(
    current_frame
)

print(
    f"Initial detections: "
    f"{len(detections)}"
)

if len(detections) == 0:

    print(
        "No drone detected on first frame."
    )

    print(
        "Press SPACE to search through "
        "the video."
    )

else:

    print(
        "Drone detection available."
    )

print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # ========================================================
    # DISPLAY CURRENT FRAME
    # ========================================================

    display = current_frame.copy()


    # ========================================================
    # NOT LOCKED
    # ========================================================

    if not target_locked:

        draw_detections(
            display,
            detections
        )

        cv2.putText(
            display,
            "PAUSED - LEFT CLICK DRONE TO LOCK",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "SPACE = next frame/search | Q = quit",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # MOUSE CLICK
        # ----------------------------------------------------

        if mouse_clicked:

            selected = select_detection(
                detections,
                mouse_x,
                mouse_y
            )

            mouse_clicked = False


            # ------------------------------------------------
            # Nothing selected
            # ------------------------------------------------

            if selected is None:

                print(
                    f"[LOCK FAILED] "
                    f"No drone near click "
                    f"({mouse_x}, {mouse_y})."
                )


            # ------------------------------------------------
            # Drone selected
            # ------------------------------------------------

            else:

                (
                    x1,
                    y1,
                    x2,
                    y2,
                    conf
                ) = selected


                candidate_box = (
                    x1,
                    y1,
                    x2,
                    y2
                )


                # --------------------------------------------
                # Create KLT features
                # --------------------------------------------

                points = create_klt_points(
                    current_gray,
                    candidate_box
                )


                if (
                    points is None
                    or
                    len(points) < 2
                ):

                    print(
                        "[LOCK FAILED] "
                        "Not enough KLT features."
                    )

                else:

                    target_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    target_conf = conf

                    prev_points = points

                    prev_gray = (
                        current_gray.copy()
                    )

                    target_locked = True

                    trajectory.clear()


                    cx = int(
                        (x1 + x2) / 2
                    )

                    cy = int(
                        (y1 + y2) / 2
                    )


                    trajectory.append(
                        (cx, cy)
                    )


                    last_center = (
                        cx,
                        cy
                    )

                    last_time = (
                        time.perf_counter()
                    )


                    vx = 0.0

                    vy = 0.0

                    speed = 0.0


                    print()

                    print(
                        "================================"
                    )

                    print(
                        "TARGET LOCKED"
                    )

                    print(
                        f"Box: "
                        f"({x1}, {y1}, "
                        f"{x2}, {y2})"
                    )

                    print(
                        f"Confidence: "
                        f"{conf:.2f}"
                    )

                    print(
                        f"KLT points: "
                        f"{len(points)}"
                    )

                    print(
                        "Press SPACE "
                        "to start tracking."
                    )

                    print(
                        "================================"
                    )

                    print()


    # ========================================================
    # TARGET LOCKED BUT PAUSED
    # ========================================================

    elif target_locked and paused:

        x1, y1, x2, y2 = target_box

        cx = int(
            (x1 + x2) / 2
        )

        cy = int(
            (y1 + y2) / 2
        )


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
            5,
            (0, 0, 255),
            -1
        )


        # Draw KLT points

        if prev_points is not None:

            for point in prev_points:

                px, py = point.ravel()

                cv2.circle(
                    display,
                    (
                        int(px),
                        int(py)
                    ),
                    3,
                    (255, 255, 0),
                    -1
                )


        cv2.putText(
            display,
            "TARGET LOCKED - PAUSED",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            "SPACE = start | R = reset | Q = quit",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    # ========================================================
    # TRACKING MODE
    # ========================================================

    elif target_locked and not paused:

        # ----------------------------------------------------
        # KLT TRACKING
        # ----------------------------------------------------

        if (
            prev_points is not None
            and
            len(prev_points) >= 2
            and
            prev_gray is not None
        ):

            new_points, status, error = (
                cv2.calcOpticalFlowPyrLK(
                    prev_gray,
                    current_gray,
                    prev_points,
                    None,
                    **lk_params
                )
            )


            if (
                new_points is not None
                and
                status is not None
            ):

                good_new = new_points[
                    status.flatten() == 1
                ]

                good_old = prev_points[
                    status.flatten() == 1
                ]


                if len(good_new) >= 2:

                    dx = float(
                        np.median(
                            good_new[:, 0]
                            -
                            good_old[:, 0]
                        )
                    )

                    dy = float(
                        np.median(
                            good_new[:, 1]
                            -
                            good_old[:, 1]
                        )
                    )


                    x1, y1, x2, y2 = (
                        target_box
                    )


                    x1 += int(
                        round(dx)
                    )

                    x2 += int(
                        round(dx)
                    )

                    y1 += int(
                        round(dy)
                    )

                    y2 += int(
                        round(dy)
                    )


                    bw = max(
                        2,
                        x2 - x1
                    )

                    bh = max(
                        2,
                        y2 - y1
                    )


                    x1 = max(
                        0,
                        min(
                            width - bw,
                            x1
                        )
                    )

                    y1 = max(
                        0,
                        min(
                            height - bh,
                            y1
                        )
                    )


                    x2 = x1 + bw

                    y2 = y1 + bh


                    target_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )


                    prev_points = (
                        good_new.reshape(
                            -1,
                            1,
                            2
                        )
                    )


                else:

                    print(
                        f"[KLT WARNING] "
                        f"Too few points at "
                        f"frame {frame_id}"
                    )

                    prev_points = None


            else:

                print(
                    f"[KLT WARNING] "
                    f"Optical flow failed "
                    f"at frame {frame_id}"
                )

                prev_points = None


        # ----------------------------------------------------
        # YOLO CORRECTION
        # ----------------------------------------------------

        if (
            frame_id % YOLO_INTERVAL == 0
        ):

            detections = get_detections(
                current_frame
            )


            if detections:

                tx1, ty1, tx2, ty2 = (
                    target_box
                )


                tcx = (
                    tx1 + tx2
                ) / 2

                tcy = (
                    ty1 + ty2
                ) / 2


                best = None

                best_score = float(
                    "inf"
                )


                for det in detections:

                    x1, y1, x2, y2, conf = (
                        det
                    )


                    cx_det = (
                        x1 + x2
                    ) / 2

                    cy_det = (
                        y1 + y2
                    ) / 2


                    distance = np.sqrt(
                        (cx_det - tcx) ** 2
                        +
                        (cy_det - tcy) ** 2
                    )


                    # Prefer detections close
                    # to the KLT target.

                    if distance < best_score:

                        best_score = distance

                        best = det


                # ------------------------------------------------
                # Accept YOLO correction
                # ------------------------------------------------

                if best is not None:

                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        conf
                    ) = best


                    target_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    target_conf = conf


                    new_points = (
                        create_klt_points(
                            current_gray,
                            target_box
                        )
                    )


                    if (
                        new_points is not None
                        and
                        len(new_points) >= 2
                    ):

                        prev_points = (
                            new_points
                        )


                    print(
                        f"[YOLO CORRECTION] "
                        f"Frame {frame_id} | "
                        f"Conf {conf:.2f} | "
                        f"Distance "
                        f"{best_score:.1f}px"
                    )


            else:

                print(
                    f"[YOLO] No detection "
                    f"at frame {frame_id}; "
                    f"KLT continues."
                )


        # ----------------------------------------------------
        # UPDATE PREVIOUS GRAY
        # ----------------------------------------------------

        prev_gray = (
            current_gray.copy()
        )


        # ----------------------------------------------------
        # TARGET CENTER
        # ----------------------------------------------------

        x1, y1, x2, y2 = target_box


        cx = int(
            (x1 + x2) / 2
        )

        cy = int(
            (y1 + y2) / 2
        )


        trajectory.append(
            (cx, cy)
        )


        # ----------------------------------------------------
        # IMAGE-SPACE VELOCITY
        # ----------------------------------------------------

        now = time.perf_counter()


        if (
            last_center is not None
            and
            last_time is not None
        ):

            dt = (
                now
                -
                last_time
            )


            if dt > 0:

                vx = (
                    cx
                    -
                    last_center[0]
                ) / dt


                vy = (
                    cy
                    -
                    last_center[1]
                ) / dt


                speed = float(
                    np.sqrt(
                        vx * vx
                        +
                        vy * vy
                    )
                )


        last_center = (
            cx,
            cy
        )

        last_time = now


        # ----------------------------------------------------
        # TARGET BOX
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
        # KLT POINTS
        # ----------------------------------------------------

        if prev_points is not None:

            for point in prev_points:

                px, py = point.ravel()

                cv2.circle(
                    display,
                    (
                        int(px),
                        int(py)
                    ),
                    3,
                    (255, 255, 0),
                    -1
                )


        # ----------------------------------------------------
        # TRAJECTORY
        # ----------------------------------------------------

        for i in range(
            1,
            len(trajectory)
        ):

            cv2.line(
                display,
                trajectory[i - 1],
                trajectory[i],
                (255, 0, 255),
                2
            )


        # ----------------------------------------------------
        # IMAGE-SPACE PREDICTION
        # ----------------------------------------------------

        prediction_time = 0.2


        pred_x = int(
            cx
            +
            vx * prediction_time
        )


        pred_y = int(
            cy
            +
            vy * prediction_time
        )


        pred_x = max(
            0,
            min(
                width - 1,
                pred_x
            )
        )


        pred_y = max(
            0,
            min(
                height - 1,
                pred_y
            )
        )


        cv2.circle(
            display,
            (
                pred_x,
                pred_y
            ),
            7,
            (0, 255, 255),
            2
        )


        # ----------------------------------------------------
        # INFORMATION
        # ----------------------------------------------------

        klt_count = (
            0
            if prev_points is None
            else len(prev_points)
        )


        cv2.putText(
            display,
            "TARGET LOCKED",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            display,
            f"KLT: {klt_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )


        cv2.putText(
            display,
            f"Confidence: {target_conf:.2f}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            f"VX: {vx:.1f} px/s",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            f"VY: {vy:.1f} px/s",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            f"Speed: {speed:.1f} px/s",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            f"Frame: "
            f"{frame_id}/{total_frames}",
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            "SPACE = pause | R = reset | Q = quit",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(20) & 0xFF


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if (
        key == ord("q")
        or
        key == 27
    ):

        print()
        print(
            "User requested exit."
        )

        quit_requested = True

        break


    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if key == ord("r"):

        target_locked = False

        target_box = None

        target_conf = 0.0

        prev_gray = None

        prev_points = None

        trajectory.clear()

        last_center = None

        last_time = None

        vx = 0.0

        vy = 0.0

        speed = 0.0

        paused = True

        mouse_clicked = False


        print()
        print(
            "[RESET] Target unlocked."
        )

        print(
            "Select another drone."
        )

        print()


    # --------------------------------------------------------
    # SPACE
    # --------------------------------------------------------

    if key == 32:

        # --------------------------------------------
        # If target isn't locked:
        # advance exactly ONE frame
        # --------------------------------------------

        if not target_locked:

            ret, next_frame = (
                cap.read()
            )


            if not ret:

                print(
                    "Video finished."
                )

                break


            frame_id += 1

            current_frame = (
                next_frame
            )


            current_gray = cv2.cvtColor(
                current_frame,
                cv2.COLOR_BGR2GRAY
            )


            print(
                f"[SEARCH] "
                f"Frame {frame_id}: "
                f"running YOLO..."
            )


            detections = (
                get_detections(
                    current_frame
                )
            )


            print(
                f"[SEARCH] "
                f"Detections: "
                f"{len(detections)}"
            )


        # --------------------------------------------
        # If target IS locked:
        # pause/resume
        # --------------------------------------------

        else:

            paused = not paused


            if paused:

                print(
                    "[PAUSED] "
                    "Tracking paused."
                )

            else:

                # Prevent time spent
                # paused from producing
                # fake velocity.

                last_time = (
                    time.perf_counter()
                )

                # Reinitialize KLT reference.

                prev_gray = (
                    current_gray.copy()
                )


                print(
                    "[RESUMED] "
                    "Tracking."
                )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL MESSAGE
# ============================================================

print()

print(
    "====================================="
)

print(
    "TRACKING FINISHED"
)

print(
    "====================================="
)

print(
    f"Output: {OUTPUT_PATH}"
)

print(
    f"Frames processed: {frame_id}"
)

print()