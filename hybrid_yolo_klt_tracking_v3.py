import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO


# ============================================================
# HYBRID YOLO + KLT TRACKER V4.1
#
# Workflow:
#
#   VIDEO PLAYING
#        |
#        | SPACE
#        v
#   VIDEO PAUSED
#        |
#        | LEFT CLICK
#        v
#   TARGET LOCKED
#        |
#        | SPACE
#        v
#   KLT + YOLO TRACKING
#
# Controls:
#   SPACE       Pause / Resume
#   LEFT CLICK  Lock drone while paused
#   R           Reset target
#   Q / ESC     Quit
# ============================================================


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r".\outputs\yolo11n_baseline\weights\best.pt"

VIDEO_PATH = r"C:\Users\thavi\Downloads\SAM ANNA.mp4"

OUTPUT_PATH = (
    r".\outputs\videos\hybrid_yolo_klt_tracking_v4_1.mp4"
)


# ============================================================
# BASIC SETTINGS
# ============================================================

CONF = 0.20

IMG_SIZE = 640

# YOLO runs periodically while tracking.
YOLO_INTERVAL = 10

# Maximum KLT feature points.
MAX_CORNERS = 80

# Minimum reliable KLT points.
MIN_KLT_POINTS = 4

# Maximum distance allowed for YOLO to correct
# the currently locked target.
MAX_CORRECTION_DISTANCE = 120


# ============================================================
# V4.1 STABILITY SETTINGS
# ============================================================

# YOLO must have stronger confidence when correcting
# an already locked target.
YOLO_CORRECTION_CONF = 0.35

# Number of consecutive weak KLT frames before recovery.
KLT_WEAK_FRAME_LIMIT = 4

# Minimum number of frames between KLT recoveries.
KLT_RECOVERY_COOLDOWN = 8

# Larger search region around tiny drone.
KLT_PAD_X = 1.2
KLT_PAD_Y = 1.5

# Maximum median optical-flow movement per frame.
MAX_FLOW_DISTANCE = 60

# Trajectory history.
TRAJECTORY_LENGTH = 100


# ============================================================
# KLT FEATURE PARAMETERS
# ============================================================

feature_params = dict(
    maxCorners=MAX_CORNERS,
    qualityLevel=0.002,
    minDistance=2,
    blockSize=5,
)


# ============================================================
# KLT OPTICAL FLOW PARAMETERS
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

if fps <= 0:
    fps = 30.0


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
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height),
)

if not writer.isOpened():

    raise RuntimeError(
        f"Could not create output:\n{OUTPUT_PATH}"
    )


# ============================================================
# HEADER
# ============================================================

print()
print("=====================================")
print("HYBRID YOLO + KLT TRACKER V4.1")
print("=====================================")
print(f"Video FPS             : {fps:.2f}")
print(f"Resolution            : {width}x{height}")
print(f"Frames                : {total_frames}")
print(f"YOLO interval         : {YOLO_INTERVAL}")
print(f"KLT max points        : {MAX_CORNERS}")
print(f"KLT minimum points    : {MIN_KLT_POINTS}")
print(f"YOLO correction conf  : {YOLO_CORRECTION_CONF}")
print("=====================================")
print("SPACE      = pause / resume")
print("LEFT CLICK = lock drone while paused")
print("R          = reset target")
print("Q / ESC    = quit")
print("=====================================")
print()


# ============================================================
# STATE
# ============================================================

paused = False

target_locked = False

target_box = None

target_conf = 0.0

prev_gray = None

prev_points = None

trajectory = deque(
    maxlen=TRAJECTORY_LENGTH
)

detections = []

current_frame = None

current_gray = None

frame_id = 0

last_center = None

last_time = None

vx = 0.0

vy = 0.0

speed = 0.0


# ============================================================
# V4.1 KLT STATE
# ============================================================

klt_weak_frames = 0

last_recovery_frame = -9999

tracking_status = "KLT STABLE"


# ============================================================
# MOUSE STATE
# ============================================================

mouse_clicked = False

mouse_x = 0

mouse_y = 0


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global mouse_clicked
    global mouse_x
    global mouse_y

    if event == cv2.EVENT_LBUTTONDOWN:

        mouse_x = x

        mouse_y = y

        mouse_clicked = True


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = (
    "HYBRID YOLO + KLT TRACKER V4.1"
)

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

    output = []

    if not results:
        return output

    result = results[0]

    if result.boxes is None:
        return output

    for box in result.boxes:

        xyxy = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
        )

        conf = float(
            box.conf[0]
            .detach()
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = map(
            int,
            xyxy
        )

        x1 = max(
            0,
            min(width - 1, x1)
        )

        y1 = max(
            0,
            min(height - 1, y1)
        )

        x2 = max(
            0,
            min(width - 1, x2)
        )

        y2 = max(
            0,
            min(height - 1, y2)
        )

        if x2 <= x1 or y2 <= y1:
            continue

        output.append(
            (
                x1,
                y1,
                x2,
                y2,
                conf
            )
        )

    return output


# ============================================================
# CREATE KLT FEATURES
# ============================================================

def create_klt_points(
    gray,
    box
):

    x1, y1, x2, y2 = box

    w = max(
        2,
        x2 - x1
    )

    h = max(
        2,
        y2 - y1
    )

    # Larger area around tiny target.
    pad_x = max(
        10,
        int(w * KLT_PAD_X)
    )

    pad_y = max(
        10,
        int(h * KLT_PAD_Y)
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

    mask = np.zeros_like(
        gray
    )

    mask[
        sy1:sy2,
        sx1:sx2
    ] = 255

    points = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        **feature_params
    )

    if points is None:
        return None

    if len(points) < 2:
        return None

    return points.astype(
        np.float32
    )


# ============================================================
# NORMALIZE POINTS
# ============================================================

def normalize_points(points):

    if points is None:
        return None

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if points.size == 0:
        return None

    return points.reshape(
        -1,
        1,
        2
    )


# ============================================================
# SELECT DETECTION
# ============================================================

def select_detection(
    detections,
    mx,
    my
):

    # --------------------------------------------------------
    # First preference: click inside detection.
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
    # Second preference: close to detection.
    # --------------------------------------------------------

    best = None

    best_distance = float(
        "inf"
    )

    tolerance = 30

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

        distance = np.sqrt(
            dx * dx +
            dy * dy
        )

        if (
            distance <= tolerance
            and
            distance < best_distance
        ):

            best_distance = distance

            best = det

    return best


# ============================================================
# BOX CENTER
# ============================================================

def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


# ============================================================
# BOX DISTANCE
# ============================================================

def center_distance(
    box_a,
    box_b
):

    ax, ay = box_center(
        box_a
    )

    bx, by = box_center(
        box_b
    )

    return float(
        np.sqrt(
            (ax - bx) ** 2
            +
            (ay - by) ** 2
        )
    )


# ============================================================
# RESET TARGET
# ============================================================

def reset_target():

    global target_locked
    global target_box
    global target_conf
    global prev_gray
    global prev_points
    global trajectory
    global last_center
    global last_time
    global vx
    global vy
    global speed
    global paused
    global klt_weak_frames
    global last_recovery_frame
    global tracking_status

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

    klt_weak_frames = 0

    last_recovery_frame = -9999

    tracking_status = "KLT STABLE"

    paused = True


# ============================================================
# FIRST FRAME
# ============================================================

ret, current_frame = cap.read()

if not ret:

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    raise RuntimeError(
        "Could not read first frame."
    )


frame_id = 1

current_gray = cv2.cvtColor(
    current_frame,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# INITIAL YOLO
# ============================================================

print(
    "[INIT] Running first YOLO detection..."
)

detections = get_detections(
    current_frame
)

print(
    f"[INIT] Detections: "
    f"{len(detections)}"
)

print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # ========================================================
    # PLAYING
    # ========================================================

    if not paused:

        ret, frame = cap.read()

        if not ret:

            print()
            print("Video finished.")

            break

        frame_id += 1

        current_frame = frame

        current_gray = cv2.cvtColor(
            current_frame,
            cv2.COLOR_BGR2GRAY
        )


        # ====================================================
        # SEARCH MODE
        # ====================================================

        if not target_locked:

            detections = get_detections(
                current_frame
            )


        # ====================================================
        # TRACKING MODE
        # ====================================================

        else:

            tracking_status = "KLT STABLE"


            # =================================================
            # KLT
            # =================================================

            if (
                prev_points is not None
                and
                len(prev_points) >= 1
                and
                prev_gray is not None
            ):

                prev_points = normalize_points(
                    prev_points
                )

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

                    status = status.reshape(-1)

                    new_flat = (
                        new_points.reshape(
                            -1,
                            2
                        )
                    )

                    old_flat = (
                        prev_points.reshape(
                            -1,
                            2
                        )
                    )

                    good_new = new_flat[
                        status == 1
                    ]

                    good_old = old_flat[
                        status == 1
                    ]


                    # -----------------------------------------
                    # Remove invalid points.
                    # -----------------------------------------

                    if len(good_new) > 0:

                        valid = (
                            np.isfinite(
                                good_new
                            ).all(axis=1)
                        )

                        good_new = (
                            good_new[
                                valid
                            ]
                        )

                        good_old = (
                            good_old[
                                valid
                            ]
                        )


                    # -----------------------------------------
                    # Enough KLT points?
                    # -----------------------------------------

                    if (
                        len(good_new)
                        >=
                        MIN_KLT_POINTS
                    ):

                        dx_values = (
                            good_new[:, 0]
                            -
                            good_old[:, 0]
                        )

                        dy_values = (
                            good_new[:, 1]
                            -
                            good_old[:, 1]
                        )

                        dx = float(
                            np.median(
                                dx_values
                            )
                        )

                        dy = float(
                            np.median(
                                dy_values
                            )
                        )


                        flow_distance = np.sqrt(
                            dx * dx +
                            dy * dy
                        )


                        # -------------------------------------
                        # Reject huge flow jumps.
                        # -------------------------------------

                        if (
                            flow_distance
                            <=
                            MAX_FLOW_DISTANCE
                        ):

                            x1, y1, x2, y2 = (
                                target_box
                            )

                            bw = x2 - x1

                            bh = y2 - y1


                            x1 += int(
                                round(dx)
                            )

                            y1 += int(
                                round(dy)
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


                            klt_weak_frames = 0

                            tracking_status = (
                                "KLT STABLE"
                            )


                        else:

                            klt_weak_frames += 1

                            tracking_status = (
                                "KLT RECOVERING"
                            )


                    else:

                        klt_weak_frames += 1

                        tracking_status = (
                            "KLT RECOVERING"
                        )


                else:

                    klt_weak_frames += 1

                    tracking_status = (
                        "KLT RECOVERING"
                    )


            else:

                klt_weak_frames += 1

                tracking_status = (
                    "KLT RECOVERING"
                )


            # =================================================
            # YOLO CORRECTION
            # =================================================

            if (
                frame_id
                %
                YOLO_INTERVAL
                ==
                0
            ):

                detections = get_detections(
                    current_frame
                )

                best = None

                best_distance = float(
                    "inf"
                )


                for det in detections:

                    x1, y1, x2, y2, conf = det

                    # -----------------------------------------
                    # Stronger confidence for correction.
                    # -----------------------------------------

                    if (
                        conf
                        <
                        YOLO_CORRECTION_CONF
                    ):
                        continue


                    det_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )


                    distance = center_distance(
                        target_box,
                        det_box
                    )


                    if (
                        distance
                        <
                        best_distance
                    ):

                        best_distance = (
                            distance
                        )

                        best = det


                # ------------------------------------------------
                # Safe YOLO correction.
                # ------------------------------------------------

                if (
                    best is not None
                    and
                    best_distance
                    <=
                    MAX_CORRECTION_DISTANCE
                ):

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
                        len(new_points)
                        >= MIN_KLT_POINTS
                    ):

                        prev_points = (
                            new_points
                        )


                    klt_weak_frames = 0

                    tracking_status = (
                        "YOLO CORRECTED"
                    )


                    print(
                        f"[YOLO] Frame {frame_id} | "
                        f"Conf {conf:.2f} | "
                        f"Distance "
                        f"{best_distance:.1f}px | "
                        f"KLT "
                        f"{0 if prev_points is None else len(prev_points)}"
                    )


                else:

                    if (
                        klt_weak_frames
                        >
                        0
                    ):

                        tracking_status = (
                            "KLT RECOVERING"
                        )

                    else:

                        tracking_status = (
                            "KLT STABLE"
                        )


            # =================================================
            # KLT RECOVERY
            # =================================================

            if (
                klt_weak_frames
                >=
                KLT_WEAK_FRAME_LIMIT
            ):

                if (
                    frame_id
                    -
                    last_recovery_frame
                    >=
                    KLT_RECOVERY_COOLDOWN
                ):

                    recovery_points = (
                        create_klt_points(
                            current_gray,
                            target_box
                        )
                    )


                    if (
                        recovery_points is not None
                        and
                        len(recovery_points)
                        >= MIN_KLT_POINTS
                    ):

                        prev_points = (
                            recovery_points
                        )

                        last_recovery_frame = (
                            frame_id
                        )

                        klt_weak_frames = 0

                        tracking_status = (
                            "KLT RECOVERED"
                        )

                        print(
                            f"[KLT] Recovery at "
                            f"frame {frame_id} | "
                            f"Points "
                            f"{len(recovery_points)}"
                        )

                    else:

                        tracking_status = (
                            "KLT WAITING FOR YOLO"
                        )


            # =================================================
            # UPDATE PREVIOUS FRAME
            # =================================================

            prev_gray = (
                current_gray.copy()
            )


            # =================================================
            # TARGET CENTER
            # =================================================

            x1, y1, x2, y2 = (
                target_box
            )

            cx = int(
                (x1 + x2) / 2
            )

            cy = int(
                (y1 + y2) / 2
            )


            trajectory.append(
                (cx, cy)
            )


            # =================================================
            # VELOCITY
            # =================================================

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


                if (
                    dt > 0
                    and
                    dt < 1.0
                ):

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
                            vx * vx +
                            vy * vy
                        )
                    )


            last_center = (
                cx,
                cy
            )

            last_time = now


    # ========================================================
    # DISPLAY
    # ========================================================

    display = current_frame.copy()


    # ========================================================
    # SEARCH DISPLAY
    # ========================================================

    if not target_locked:

        for i, det in enumerate(
            detections
        ):

            x1, y1, x2, y2, conf = det


            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                2
            )


            cv2.putText(
                display,
                f"DRONE {i + 1} {conf:.2f}",
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


        if paused:

            cv2.putText(
                display,
                "PAUSED - CLICK DRONE",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                "CLICK = lock | SPACE = resume",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                display,
                "YOLO SEARCH",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )

            cv2.putText(
                display,
                "SPACE = pause | Q = quit",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


    # ========================================================
    # TRACKING DISPLAY
    # ========================================================

    else:

        x1, y1, x2, y2 = (
            target_box
        )


        cx = int(
            (x1 + x2) / 2
        )

        cy = int(
            (y1 + y2) / 2
        )


        # ----------------------------------------------------
        # Target box
        # ----------------------------------------------------

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # Target center
        # ----------------------------------------------------

        cv2.circle(
            display,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )


        # ----------------------------------------------------
        # KLT points
        # ----------------------------------------------------

        if prev_points is not None:

            points_display = (
                prev_points.reshape(
                    -1,
                    2
                )
            )


            for px, py in points_display:

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
        # Trajectory
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
        # Short-term prediction
        # ----------------------------------------------------

        prediction_time = 0.2

        pred_x = int(
            cx +
            vx * prediction_time
        )

        pred_y = int(
            cy +
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
        # Status
        # ----------------------------------------------------

        cv2.putText(
            display,
            tracking_status,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # KLT count
        # ----------------------------------------------------

        klt_count = (
            0
            if prev_points is None
            else len(prev_points)
        )


        cv2.putText(
            display,
            f"KLT points: {klt_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"YOLO confidence: {target_conf:.2f}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Velocity
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Frame
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Frame: {frame_id}/{total_frames}",
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Controls
        # ----------------------------------------------------

        cv2.putText(
            display,
            "SPACE=pause | R=reset | Q=quit",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display
    )


    # ========================================================
    # WRITE ONLY NEW VIDEO FRAMES
    # ========================================================

    if not paused:

        writer.write(
            display
        )


    # ========================================================
    # KEY INPUT
    # ========================================================

    if paused:

        key = cv2.waitKey(30) & 0xFF

    else:

        key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # QUIT
    # ========================================================

    if (
        key == ord("q")
        or
        key == 27
    ):

        print()
        print(
            "[EXIT] User requested exit."
        )

        break


    # ========================================================
    # RESET
    # ========================================================

    if key == ord("r"):

        reset_target()

        print()
        print(
            "[RESET] Target unlocked."
        )

        print(
            "[RESET] Video paused."
        )

        print(
            "[RESET] Click a drone."
        )

        print(
            "[RESET] Press SPACE to resume."
        )

        print()


    # ========================================================
    # SPACE
    # ========================================================

    if key == 32:

        if paused:

            paused = False

            last_time = (
                time.perf_counter()
            )

            if target_locked:

                prev_gray = (
                    current_gray.copy()
                )

            print(
                f"[PLAY] Resumed at "
                f"frame {frame_id}"
            )

        else:

            paused = True

            print(
                f"[PAUSE] Frozen at "
                f"frame {frame_id}"
            )


    # ========================================================
    # MOUSE LOCK
    # ========================================================

    if mouse_clicked:

        if (
            paused
            and
            not target_locked
        ):

            selected = select_detection(
                detections,
                mouse_x,
                mouse_y
            )


            if selected is None:

                print()
                print(
                    "[LOCK] No detection near "
                    f"({mouse_x}, {mouse_y})"
                )

                print(
                    "[LOCK] Click directly "
                    "inside the orange box."
                )


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
                # Initialize KLT.
                # --------------------------------------------

                points = create_klt_points(
                    current_gray,
                    candidate_box
                )


                point_count = (
                    0
                    if points is None
                    else len(points)
                )


                if (
                    point_count
                    <
                    MIN_KLT_POINTS
                ):

                    print()
                    print(
                        "[LOCK] Too few KLT features."
                    )

                    print(
                        f"[LOCK] Found: "
                        f"{point_count}"
                    )

                    print(
                        "[LOCK] Try another frame."
                    )


                else:

                    # ----------------------------------------
                    # Lock.
                    # ----------------------------------------

                    target_locked = True

                    target_box = (
                        candidate_box
                    )

                    target_conf = conf

                    prev_points = (
                        normalize_points(
                            points
                        )
                    )

                    prev_gray = (
                        current_gray.copy()
                    )


                    # ----------------------------------------
                    # Reset tracking state.
                    # ----------------------------------------

                    klt_weak_frames = 0

                    last_recovery_frame = (
                        -9999
                    )

                    tracking_status = (
                        "TARGET LOCKED"
                    )


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
                        "[LOCK SUCCESS]"
                    )

                    print(
                        f"Frame       : {frame_id}"
                    )

                    print(
                        f"Box         : "
                        f"({x1}, {y1}, "
                        f"{x2}, {y2})"
                    )

                    print(
                        f"Confidence  : "
                        f"{conf:.2f}"
                    )

                    print(
                        f"KLT points  : "
                        f"{point_count}"
                    )

                    print(
                        "State       : PAUSED"
                    )

                    print(
                        "Action      : Press SPACE "
                        "to start tracking"
                    )

                    print(
                        "================================"
                    )

                    print()


        # Consume click.

        mouse_clicked = False


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL
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