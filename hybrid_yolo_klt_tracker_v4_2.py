import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r".\outputs\yolo11n_baseline\weights\best.pt"
VIDEO_PATH = r"C:\Users\thavi\Downloads\SAM ANNA.mp4"

OUTPUT_PATH = r".\outputs\videos\hybrid_yolo_klt_tracking_v4_2.mp4"

IMG_SIZE = 640

# Detection
YOLO_CONF = 0.25
YOLO_INTERVAL = 5

# KLT
MAX_KLT_POINTS = 30
MIN_KLT_POINTS = 4

# KLT quality
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 3
BLOCK_SIZE = 7

# Maximum allowed movement of the KLT box
MAX_KLT_MOVE = 25

# Maximum disagreement between KLT and YOLO centers
MAX_YOLO_KLT_DISTANCE = 35

# Small padding around target when searching for KLT points
KLT_PADDING = 4

# Trajectory
TRAJECTORY_LENGTH = 80

# Prediction
PREDICTION_SECONDS = 0.15


# ============================================================
# GLOBAL STATE
# ============================================================

mouse_x = 0
mouse_y = 0
mouse_clicked = False


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_clicked

    mouse_x = x
    mouse_y = y

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_drone(model, frame):
    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=YOLO_CONF,
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

        x1, y1, x2, y2 = map(int, xyxy)

        conf = float(box.conf[0])

        detections.append(
            (x1, y1, x2, y2, conf)
        )

    return detections


# ============================================================
# SELECT DETECTION UNDER MOUSE
# ============================================================

def select_detection(detections, x, y):

    selected = None

    for det in detections:

        x1, y1, x2, y2, conf = det

        if x1 <= x <= x2 and y1 <= y <= y2:

            selected = det
            break

    return selected


# ============================================================
# CREATE KLT POINTS
# ============================================================

def create_klt_points(gray, box):

    x1, y1, x2, y2 = box

    h, w = gray.shape

    x1 = max(0, x1 + KLT_PADDING)
    y1 = max(0, y1 + KLT_PADDING)

    x2 = min(w - 1, x2 - KLT_PADDING)
    y2 = min(h - 1, y2 - KLT_PADDING)

    if x2 <= x1 or y2 <= y1:
        return None

    mask = np.zeros_like(gray)

    mask[y1:y2, x1:x2] = 255

    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_KLT_POINTS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE,
        mask=mask
    )

    return points


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
# DISTANCE
# ============================================================

def center_distance(box1, box2):

    c1 = box_center(box1)
    c2 = box_center(box2)

    return np.sqrt(
        (c1[0] - c2[0]) ** 2 +
        (c1[1] - c2[1]) ** 2
    )


# ============================================================
# UPDATE BOX USING KLT
# ============================================================

def update_box_with_klt(old_box, old_points, new_points):

    if old_points is None or new_points is None:
        return old_box, None

    if len(old_points) < MIN_KLT_POINTS:
        return old_box, None

    old_points = old_points.reshape(-1, 2)
    new_points = new_points.reshape(-1, 2)

    motions = new_points - old_points

    # --------------------------------------------------------
    # Remove extreme optical-flow movements
    # --------------------------------------------------------

    motion_magnitude = np.linalg.norm(motions, axis=1)

    valid = motion_magnitude < MAX_KLT_MOVE

    if np.sum(valid) < MIN_KLT_POINTS:
        return old_box, None

    motions = motions[valid]
    new_points = new_points[valid]

    # --------------------------------------------------------
    # MEDIAN MOTION
    # --------------------------------------------------------

    dx = float(np.median(motions[:, 0]))
    dy = float(np.median(motions[:, 1]))

    # --------------------------------------------------------
    # Limit one-frame movement
    # --------------------------------------------------------

    dx = np.clip(dx, -MAX_KLT_MOVE, MAX_KLT_MOVE)
    dy = np.clip(dy, -MAX_KLT_MOVE, MAX_KLT_MOVE)

    x1, y1, x2, y2 = old_box

    new_box = (
        int(x1 + dx),
        int(y1 + dy),
        int(x2 + dx),
        int(y2 + dy)
    )

    return new_box, new_points.reshape(-1, 1, 2)


# ============================================================
# DRAW BOX
# ============================================================

def draw_box(frame, box, color, thickness=2):

    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global mouse_clicked

    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    print("YOLO model loaded.")

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():

        print("ERROR: Could not open video.")

        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print()
    print("=====================================")
    print("HYBRID YOLO + KLT TRACKER V4.2")
    print("=====================================")
    print(f"Video FPS             : {fps:.2f}")
    print(f"Resolution            : {width}x{height}")
    print(f"Frames                : {total_frames}")
    print(f"YOLO interval         : {YOLO_INTERVAL}")
    print(f"KLT max points        : {MAX_KLT_POINTS}")
    print(f"KLT minimum points    : {MIN_KLT_POINTS}")
    print(f"YOLO correction conf  : {YOLO_CONF}")
    print("=====================================")
    print("SPACE      = pause / resume")
    print("LEFT CLICK = lock drone while paused")
    print("R          = reset target")
    print("Q / ESC    = quit")
    print("=====================================")

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------------
    # INITIAL FRAME
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read video.")

        cap.release()
        writer.release()

        return

    frame_number = 1

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    paused = True
    target_locked = False

    target_box = None

    klt_points = None
    previous_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    trajectory = deque(
        maxlen=TRAJECTORY_LENGTH
    )

    velocity_x = 0.0
    velocity_y = 0.0

    # --------------------------------------------------------
    # WINDOW
    # --------------------------------------------------------

    window_name = "YOLO + KLT V4.2"

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while True:

        # ====================================================
        # PAUSED / SELECTION MODE
        # ====================================================

        if paused:

            display = frame.copy()

            # -----------------------------------------------
            # If not locked, show YOLO detections
            # -----------------------------------------------

            if not target_locked:

                detections = detect_drone(
                    model,
                    frame
                )

                for det in detections:

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
                        f"Drone {conf:.2f}",
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 165, 255),
                        2
                    )

                cv2.putText(
                    display,
                    "PAUSED - LEFT CLICK DRONE TO LOCK",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            # -----------------------------------------------
            # Locked but waiting for SPACE
            # -----------------------------------------------

            else:

                draw_box(
                    display,
                    target_box,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    "TARGET LOCKED - PRESS SPACE",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                if klt_points is not None:

                    for p in klt_points.reshape(-1, 2):

                        px, py = map(int, p)

                        cv2.circle(
                            display,
                            (px, py),
                            2,
                            (255, 255, 0),
                            -1
                        )

            # -----------------------------------------------
            # Mouse click
            # -----------------------------------------------

            if mouse_clicked and not target_locked:

                mouse_clicked = False

                detections = detect_drone(
                    model,
                    frame
                )

                selected = select_detection(
                    detections,
                    mouse_x,
                    mouse_y
                )

                if selected is not None:

                    x1, y1, x2, y2, conf = selected

                    target_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    gray = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2GRAY
                    )

                    klt_points = create_klt_points(
                        gray,
                        target_box
                    )

                    if (
                        klt_points is not None
                        and len(klt_points) >= MIN_KLT_POINTS
                    ):

                        target_locked = True

                        print()
                        print("================================")
                        print("[LOCK SUCCESS]")
                        print(f"Frame       : {frame_number}")
                        print(f"Box         : {target_box}")
                        print(f"Confidence  : {conf:.2f}")
                        print(
                            f"KLT points  : {len(klt_points)}"
                        )
                        print("State       : PAUSED")
                        print(
                            "Action      : Press SPACE to start tracking"
                        )
                        print("================================")

                    else:

                        print(
                            "[LOCK FAILED] Not enough KLT points."
                        )

            # -----------------------------------------------
            # Keyboard
            # -----------------------------------------------

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == 27:

                break

            elif key == ord("r"):

                target_locked = False
                target_box = None
                klt_points = None
                trajectory.clear()

                print("[RESET] Target reset.")

            elif key == 32 and target_locked:

                paused = False

                print(
                    f"[PLAY] Resumed at frame {frame_number}"
                )

            cv2.imshow(
                window_name,
                display
            )

            continue

        # ====================================================
        # TRACKING MODE
        # ====================================================

        ret, next_frame = cap.read()

        if not ret:

            print()
            print("Video finished.")

            break

        frame = next_frame
        frame_number += 1

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # KLT TRACKING
        # ----------------------------------------------------

        if klt_points is not None and len(klt_points) >= MIN_KLT_POINTS:

            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                gray,
                klt_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS |
                    cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01
                )
            )

            if next_points is not None:

                good_old = klt_points[
                    status.flatten() == 1
                ]

                good_new = next_points[
                    status.flatten() == 1
                ]

                if len(good_new) >= MIN_KLT_POINTS:

                    old_box = target_box

                    updated_box, filtered_points = (
                        update_box_with_klt(
                            target_box,
                            good_old,
                            good_new
                        )
                    )

                    if filtered_points is not None:

                        target_box = updated_box

                        klt_points = filtered_points

                        # ----------------------------------
                        # Velocity estimate
                        # ----------------------------------

                        dx = (
                            target_box[0]
                            - old_box[0]
                        )

                        dy = (
                            target_box[1]
                            - old_box[1]
                        )

                        velocity_x = (
                            0.7 * velocity_x
                            + 0.3 * dx * fps
                        )

                        velocity_y = (
                            0.7 * velocity_y
                            + 0.3 * dy * fps
                        )

        # ----------------------------------------------------
        # YOLO CORRECTION
        # ----------------------------------------------------

        if frame_number % YOLO_INTERVAL == 0:

            detections = detect_drone(
                model,
                frame
            )

            best_detection = None
            best_distance = float("inf")

            if target_box is not None:

                predicted_center = box_center(
                    target_box
                )

                for det in detections:

                    x1, y1, x2, y2, conf = det

                    candidate_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    candidate_center = box_center(
                        candidate_box
                    )

                    distance = np.sqrt(
                        (
                            predicted_center[0]
                            - candidate_center[0]
                        ) ** 2
                        +
                        (
                            predicted_center[1]
                            - candidate_center[1]
                        ) ** 2
                    )

                    if distance < best_distance:

                        best_distance = distance
                        best_detection = det

            # ------------------------------------------------
            # YOLO FOUND TARGET NEAR KLT
            # ------------------------------------------------

            if (
                best_detection is not None
                and best_distance <= MAX_YOLO_KLT_DISTANCE
            ):

                x1, y1, x2, y2, conf = (
                    best_detection
                )

                yolo_box = (
                    x1,
                    y1,
                    x2,
                    y2
                )

                old_box = target_box

                # --------------------------------------------
                # YOLO becomes authoritative
                # --------------------------------------------

                target_box = yolo_box

                # --------------------------------------------
                # Rebuild KLT points from corrected box
                # --------------------------------------------

                klt_points = create_klt_points(
                    gray,
                    target_box
                )

                # --------------------------------------------
                # Velocity from YOLO correction
                # --------------------------------------------

                if old_box is not None:

                    old_center = box_center(
                        old_box
                    )

                    new_center = box_center(
                        target_box
                    )

                    dx = (
                        new_center[0]
                        - old_center[0]
                    )

                    dy = (
                        new_center[1]
                        - old_center[1]
                    )

                    velocity_x = (
                        0.5 * velocity_x
                        + 0.5 * dx * fps
                    )

                    velocity_y = (
                        0.5 * velocity_y
                        + 0.5 * dy * fps
                    )

                print(
                    f"[YOLO] Frame {frame_number} | "
                    f"Conf {conf:.2f} | "
                    f"Distance {best_distance:.1f}px | "
                    f"KLT "
                    f"{len(klt_points) if klt_points is not None else 0}"
                )

            # ------------------------------------------------
            # YOLO DISAGREES
            # ------------------------------------------------

            elif best_detection is not None:

                print(
                    f"[YOLO REJECT] Frame {frame_number} | "
                    f"Distance {best_distance:.1f}px"
                )

                # Do NOT move the box toward the
                # distant detection.
                #
                # KLT remains temporarily in control.

        # ----------------------------------------------------
        # KLT POINT RECOVERY
        # ----------------------------------------------------

        # Only recover if points are genuinely low.
        # Do NOT refill continuously.

        if (
            klt_points is None
            or len(klt_points) < MIN_KLT_POINTS
        ):

            new_points = create_klt_points(
                gray,
                target_box
            )

            if (
                new_points is not None
                and len(new_points) >= MIN_KLT_POINTS
            ):

                klt_points = new_points

                print(
                    f"[KLT RECOVERY] Frame "
                    f"{frame_number} | "
                    f"Points {len(klt_points)}"
                )

        # ----------------------------------------------------
        # TRAJECTORY
        # ----------------------------------------------------

        center = box_center(target_box)

        trajectory.append(center)

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        display = frame.copy()

        # Target box
        draw_box(
            display,
            target_box,
            (0, 255, 0),
            2
        )

        # KLT points
        if klt_points is not None:

            for p in klt_points.reshape(-1, 2):

                px, py = map(int, p)

                cv2.circle(
                    display,
                    (px, py),
                    2,
                    (255, 255, 0),
                    -1
                )

        # ----------------------------------------------------
        # TRAJECTORY
        # ----------------------------------------------------

        if len(trajectory) >= 2:

            for i in range(1, len(trajectory)):

                p1 = tuple(
                    map(int, trajectory[i - 1])
                )

                p2 = tuple(
                    map(int, trajectory[i])
                )

                cv2.line(
                    display,
                    p1,
                    p2,
                    (0, 255, 255),
                    2
                )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        cx, cy = center

        predicted_x = int(
            cx + velocity_x * PREDICTION_SECONDS
        )

        predicted_y = int(
            cy + velocity_y * PREDICTION_SECONDS
        )

        cv2.circle(
            display,
            (predicted_x, predicted_y),
            5,
            (0, 0, 255),
            -1
        )

        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------

        cv2.putText(
            display,
            "TRACKING",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Frame: {frame_number}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"KLT points: "
            f"{len(klt_points) if klt_points is not None else 0}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Velocity: "
            f"{velocity_x:.1f}, {velocity_y:.1f} px/s",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "SPACE=pause  R=reset  Q=quit",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # WRITE FRAME
        # ----------------------------------------------------

        writer.write(display)

        cv2.imshow(
            window_name,
            display
        )

        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:

            break

        elif key == 32:

            paused = True

            print(
                f"[PAUSE] Frozen at frame "
                f"{frame_number}"
            )

        elif key == ord("r"):

            target_locked = False
            paused = True

            target_box = None
            klt_points = None

            trajectory.clear()

            print(
                "[RESET] Target reset. "
                "Click a drone to lock."
            )

        previous_gray = gray

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    cv2.destroyAllWindows()

    print()
    print("=====================================")
    print("TRACKING FINISHED")
    print("=====================================")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Frames processed: {frame_number}")


if __name__ == "__main__":
    main()