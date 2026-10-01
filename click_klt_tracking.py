import cv2
import numpy as np
from ultralytics import YOLO

MODEL = r".\outputs\yolo11n_baseline\weights\best.pt"
VIDEO = r"C:\Users\thavi\Downloads\SAM ANNA.mp4"

CONF = 0.20
IMG_SIZE = 640

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open video")

WINDOW = "YOLO + KLT Target Lock"

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

paused = False
locked = False

frame = None
gray = None
previous_gray = None

points = None
target_box = None

detections = []
confidences = []

frame_count = 0


def expanded_box(box, frame_shape, scale=1.8):

    h, w = frame_shape[:2]

    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    nx1 = int(max(0, cx - bw / 2))
    ny1 = int(max(0, cy - bh / 2))

    nx2 = int(min(w - 1, cx + bw / 2))
    ny2 = int(min(h - 1, cy + bh / 2))

    return nx1, ny1, nx2, ny2


def get_features(gray_image, box):

    # Expand the search area.
    search_box = expanded_box(
        box,
        gray_image.shape,
        scale=2.0
    )

    x1, y1, x2, y2 = search_box

    mask = np.zeros_like(gray_image)

    cv2.rectangle(
        mask,
        (x1, y1),
        (x2, y2),
        255,
        -1
    )

    pts = cv2.goodFeaturesToTrack(
        gray_image,
        mask=mask,
        maxCorners=80,
        qualityLevel=0.001,
        minDistance=2,
        blockSize=3
    )

    return pts


def run_detection():

    global detections
    global confidences

    result = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        device=0,
        verbose=False
    )[0]

    detections = []
    confidences = []

    if result.boxes is None:
        return

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for box, conf in zip(boxes, confs):

        x1, y1, x2, y2 = map(
            int,
            box
        )

        detections.append(
            (x1, y1, x2, y2)
        )

        confidences.append(
            float(conf)
        )


def lock_target(index):

    global locked
    global points
    global target_box

    if index < 0 or index >= len(detections):
        return

    box = detections[index]

    new_points = get_features(
        gray,
        box
    )

    if new_points is None or len(new_points) < 1:

        print(
            "[LOCK FAILED] "
            "No usable KLT features."
        )

        return

    target_box = tuple(
        map(int, box)
    )

    points = new_points

    locked = True

    print()
    print("==============================")
    print("TARGET LOCKED")
    print("Box:", target_box)
    print("KLT points:", len(points))
    print("==============================")
    print()


def mouse_callback(event, x, y, flags, param):

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if not paused:
        print(
            "[INFO] Press SPACE first "
            "to pause."
        )
        return

    for i, box in enumerate(detections):

        x1, y1, x2, y2 = box

        if (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        ):

            print(
                f"[INFO] Selecting detection "
                f"{i + 1}"
            )

            lock_target(i)

            return

    print(
        "[INFO] Click directly inside "
        "an orange YOLO box."
    )


cv2.namedWindow(
    WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.setMouseCallback(
    WINDOW,
    mouse_callback
)


while True:

    if not paused:

        ret, new_frame = cap.read()

        if not ret:
            break

        frame = new_frame

        frame_count += 1

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        if not locked:

            run_detection()

        else:

            if (
                previous_gray is not None
                and
                points is not None
            ):

                new_points, status, error = (
                    cv2.calcOpticalFlowPyrLK(
                        previous_gray,
                        gray,
                        points,
                        None,
                        **lk_params
                    )
                )

                if new_points is not None:

                    good_new = new_points[
                        status.flatten() == 1
                    ]

                    good_old = points[
                        status.flatten() == 1
                    ]

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

                        x1, y1, x2, y2 = target_box

                        target_box = (
                            int(x1 + dx),
                            int(y1 + dy),
                            int(x2 + dx),
                            int(y2 + dy),
                        )

                        points = (
                            good_new
                            .reshape(-1, 1, 2)
                        )

                    else:

                        print(
                            "[KLT] Lost features. "
                            "Trying YOLO recovery..."
                        )

                        locked = False
                        points = None

                        run_detection()

        previous_gray = gray.copy()

    # ------------------------------------------------
    # DRAW
    # ------------------------------------------------

    if frame is None:
        continue

    display = frame.copy()

    if not locked:

        # YOLO detections
        for i, (box, conf) in enumerate(
            zip(detections, confidences)
        ):

            x1, y1, x2, y2 = box

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                2
            )

            label = (
                f"{i + 1}: DRONE "
                f"{conf:.2f}"
            )

            cv2.putText(
                display,
                label,
                (x1, max(25, y1 - 8)),
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
                "CLICK DRONE BOX TO LOCK",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                "Keys 1-9 = select detection",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    else:

        x1, y1, x2, y2 = target_box

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        if points is not None:

            for p in points.reshape(-1, 2):

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

        cx = int(
            (x1 + x2) / 2
        )

        cy = int(
            (y1 + y2) / 2
        )

        cv2.circle(
            display,
            (cx, cy),
            6,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            display,
            "TARGET LOCKED",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"KLT FEATURES: {len(points)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "SPACE = PAUSE | Q = QUIT",
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        WINDOW,
        display
    )

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break

    # Pause / resume
    if key == ord(" "):

        paused = not paused

        if paused:

            print(
                "[PAUSED] "
                "Click the drone or press 1-9."
            )

        else:

            print(
                "[RESUMED] Tracking."
            )

    # Number-key target selection
    if paused:

        if ord("1") <= key <= ord("9"):

            index = key - ord("1")

            if index < len(detections):

                print(
                    f"[INFO] "
                    f"Selecting detection "
                    f"{index + 1}"
                )

                lock_target(index)

cap.release()

cv2.destroyAllWindows()

print("Finished.")