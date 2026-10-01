from collections import defaultdict, deque
from pathlib import Path
import math
import cv2
from ultralytics import YOLO

# YOLO11n + ByteTrack visual tracking simulation.
MODEL = Path(r".\outputs\yolo11n_baseline\weights\best.pt")
SOURCE = Path(r"C:\Users\thavi\Downloads\SAM ANNA.mp4")
OUTPUT_DIR = Path(r".\outputs\videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTPUT_DIR / "drone_tracking_simulation.mp4"

IMG_SIZE = 640
CONF = 0.20
TRACKER = "bytetrack.yaml"
TRAIL_LENGTH = 40
PREDICT_SECONDS = 0.5

model = YOLO(str(MODEL))

cap = cv2.VideoCapture(str(SOURCE))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {SOURCE}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUTPUT), fourcc, fps, (width, height))

trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
previous_centers = {}
previous_time = {}
frame_number = 0

for result in model.track(
    source=str(SOURCE),
    imgsz=IMG_SIZE,
    conf=CONF,
    tracker=TRACKER,
    device=0,
    stream=True,
    persist=True,
    verbose=False,
):
    frame_number += 1
    frame = result.orig_img.copy()

    cv2.rectangle(frame, (0, 0), (width, 48), (0, 0, 0), -1)
    cv2.putText(
        frame, "DRONE TRACKING SIMULATION | YOLO11n + ByteTrack",
        (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2, cv2.LINE_AA
    )

    active_ids = set()

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else list(range(len(xyxy)))

        for box, confidence, track_id in zip(xyxy, confs, ids):
            track_id = int(track_id)
            active_ids.add(track_id)

            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            trails[track_id].append((cx, cy))

            now = frame_number / fps
            vx = vy = 0.0
            if track_id in previous_centers:
                old_x, old_y = previous_centers[track_id]
                dt = max(now - previous_time[track_id], 1.0 / fps)
                vx = (cx - old_x) / dt
                vy = (cy - old_y) / dt

            previous_centers[track_id] = (cx, cy)
            previous_time[track_id] = now

            speed = math.hypot(vx, vy)
            pred_x = int(cx + vx * PREDICT_SECONDS)
            pred_y = int(cy + vy * PREDICT_SECONDS)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

            pts = list(trails[track_id])
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (255, 180, 0), 2)

            cv2.line(frame, (cx, cy), (pred_x, pred_y), (255, 0, 255), 2)
            cv2.circle(frame, (pred_x, pred_y), 7, (255, 0, 255), 2)

            label = f"DRONE ID {track_id} | {confidence:.2f}"
            cv2.rectangle(frame, (x1, max(48, y1 - 30)),
                          (min(width - 1, x1 + 220), max(48, y1)), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 5, max(69, y1 - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)

            # State panel
            px, py = 12, 62
            pw, ph = 310, 132
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 0), 1)

            lines = [
                f"TARGET ID     : {track_id}",
                f"CONFIDENCE    : {confidence:.2f}",
                f"CENTER        : ({cx}, {cy})",
                f"VELOCITY      : ({vx:.1f}, {vy:.1f}) px/s",
                f"SPEED         : {speed:.1f} px/s",
                "STATUS        : TRACKING",
            ]
            for j, line in enumerate(lines):
                cv2.putText(frame, line, (22, 84 + j * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                            (255, 255, 255), 1, cv2.LINE_AA)

    if not active_ids:
        cv2.putText(frame, "TARGET STATUS: SEARCHING",
                    (12, height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 180, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Frame: {frame_number}",
                (width - 150, height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (255, 255, 255), 1, cv2.LINE_AA)

    writer.write(frame)

cap.release()
writer.release()

print("Tracking simulation complete.")
print(f"Output: {OUTPUT.resolve()}")
print(f"Frames: {frame_number}")
print(f"FPS: {fps:.2f}")
print(f"Resolution: {width}x{height}")
