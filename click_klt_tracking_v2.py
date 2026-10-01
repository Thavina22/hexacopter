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
    raise RuntimeError(f"Could not open video: {VIDEO}")

WINDOW = "Click-to-Lock + KLT"

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
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


def get_features(gray_image, box):
    x1, y1, x2, y2 = box

    mask = np.zeros_like(gray_image)

    # Slightly shrink the region so we don't select the box boundary.
    x1 += 3
    y1 += 3
    x2 -= 3
    y2 -= 3

    if x2 <= x1 or y2 <= y1:
        return None

    cv2.rectangle(
        mask,
        (x1, y1),
        (x2, y2),
        255,
        -1,
    )

    return cv2.goodFeaturesToTrack(
        gray_image,
        mask=mask,
        maxCorners=100,
        qualityLevel=0.005,
        minDistance=3,
        blockSize=5,
    )


def mouse_callback(event, x, y, flags, param):
    global locked
    global points
    global target_box

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Only allow locking while paused.
    if not paused:
        print("[INFO] Press SPACE first to pause.")
        return

    for box in detections:

        x1, y1, x2, y2 = box

        if x1 <= x <= x2 and y1 <= y <= y2:

            new_points = get_features(gray, box)

            if new_points is None or len(new_points) < 3:
                print("[LOCK FAILED] Not enough KLT features.")
                return

            target_box = tuple(map(int, box))
            points = new_points
            locked = True

            print()
            print("================================")
            print("TARGET LOCKED")
            print(f"Box: {target_box}")
            print(f"KLT points: {len(points)}")
            print("Press SPACE to resume tracking.")
            print("================================")
            print()

            return

    print("[INFO] Click directly inside an orange YOLO box.")


cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW, mouse_callback)

while True:

    if not paused:

        ret, new_frame = cap.read()

        if not ret:
            break

        frame = new_frame

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        # ------------------------------------------
        # YOLO detection before target is locked
        # ------------------------------------------

        if not locked:

            result = model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF,
                device=0,
                verbose=False,
            )[0]

            detections = []

            if result.boxes is not None:

                for box in result.boxes.xyxy.cpu().numpy():

                    x1, y1, x2, y2 = map(
                        int,
                        box,
                    )

                    detections.append(
                        (x1, y1, x2, y2)
                    )

        # ------------------------------------------
        # KLT tracking after target is locked
        # ------------------------------------------

        elif previous_gray is not None and points is not None:

            new_points, status, error = (
                cv2.calcOpticalFlowPyrLK(
                    previous_gray,
                    gray,
                    points,
                    None,
                    **lk_params,
                )
            )

            if new_points is not None:

                good_new = new_points[
                    status.flatten() == 1
                ]

                good_old = points[
                    status.flatten() == 1
                ]

                if len(good_new) >= 3:

                    dx = np.median(
                        good_new[:, 0]
                        - good_old[:, 0]
                    )

                    dy = np.median(
                        good_new[:, 1]
                        - good_old[:, 1]
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
                        "[LOST] Too few KLT points."
                    )

                    locked = False
                    points = None

        previous_gray = gray.copy()

    # ==============================================
    # DRAW
    # ==============================================

    display = frame.copy()

    if not locked:

        # Draw YOLO detections.
        for x1, y1, x2, y2 in detections:

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                2,
            )

            cv2.putText(
                display,
                "CLICK TO LOCK",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
            )

        cv2.putText(
            display,
            "SPACE = PAUSE | Click drone while paused",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    else:

        x1, y1, x2, y2 = target_box

        # Locked target.
        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        # KLT feature points.
        if points is not None:

            for p in points.reshape(-1, 2):

                px, py = map(
                    int,
                    p,
                )

                cv2.circle(
                    display,
                    (px, py),
                    3,
                    (255, 0, 0),
                    -1,
                )

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.circle(
            display,
            (cx, cy),
            6,
            (0, 255, 255),
            -1,
        )

        cv2.putText(
            display,
            "TARGET LOCKED",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            f"KLT FEATURES: {len(points) if points is not None else 0}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            "SPACE = PAUSE | Q = QUIT",
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    cv2.imshow(
        WINDOW,
        display,
    )

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break

    if key == ord(" "):

        paused = not paused

        if paused:
            print("[PAUSED] Click the drone to lock it.")
        else:
            print("[RESUMED] Tracking.")

cap.release()
cv2.destroyAllWindows()

print("Finished.")