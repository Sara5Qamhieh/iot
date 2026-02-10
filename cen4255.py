import time
import serial
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

# ---------------- Serial ----------------
SERIAL_PORT = "/dev/ttyUSB0"   # or /dev/ttyACM0
BAUD_RATE = 115200

# ---------------- YOLO ----------------
MODEL_PATH = "yolo11n.pt"      # or yolov8n.pt
IMG_SIZE = 640
CONF_THRES = 0.30              # a bit lower helps
STABLE_FRAMES_REQUIRED = 2     # make it easier to trigger on Pi
COOLDOWN_SECONDS = 2.0

RECYCLABLE_CLASSES = {"bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl"}
IGNORE_CLASSES = {"person"}

# ---------------- Camera / ROI ----------------
FRAME_W, FRAME_H = 1280, 960   # 4:3
# Start BIG to avoid “partial bottle” problems. Tighten later.
ROI_X1, ROI_Y1 = 0.15, 0.20
ROI_X2, ROI_Y2 = 0.85, 0.95

# If at least this fraction of bbox overlaps ROI, we accept it
ROI_OVERLAP_MIN = 0.15

# ---------------- One-time trigger latch ----------------
MISSING_FRAMES_TO_RESET = 10


def parse_fill_data(line):
    try:
        parts = line.strip().split("|")
        b1 = parts[0].strip()
        b2 = parts[1].strip()
        b1_cm = float(b1.split(",")[0].split(":")[1])
        b1_full = int(b1.split(",")[1].split(":")[1])
        b2_cm = float(b2.split(",")[0].split(":")[1])
        b2_full = int(b2.split(",")[1].split(":")[1])
        return b1_cm, b1_full, b2_cm, b2_full
    except:
        return None


def rect_intersection_area(a, b):
    # a,b = (x1,y1,x2,y2)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def rect_area(r):
    return float(max(0, r[2] - r[0]) * max(0, r[3] - r[1]))


def bbox_overlaps_roi(bbox, roi):
    inter = rect_intersection_area(bbox, roi)
    a = rect_area(bbox)
    if a <= 0:
        return False
    return (inter / a) >= ROI_OVERLAP_MIN


def pick_decision(result, names, roi):
    """
    Returns: (decision 'L'/'R', label, conf) or (None,None,0)
    Logic:
      - ignore 'person'
      - consider only boxes that overlap ROI enough
      - if any recyclable class exists -> choose best recyclable
      - else choose best other object
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None, None, 0.0

    candidates_recy = []
    candidates_other = []

    for b in result.boxes:
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        name = names.get(cls_id, str(cls_id)).strip().lower()

        if name in IGNORE_CLASSES:
            continue
        if conf < CONF_THRES:
            continue

        x1, y1, x2, y2 = b.xyxy[0].tolist()
        bbox = (x1, y1, x2, y2)

        if not bbox_overlaps_roi(bbox, roi):
            continue

        if name in RECYCLABLE_CLASSES:
            candidates_recy.append((conf, name))
        else:
            candidates_other.append((conf, name))

    if candidates_recy:
        conf, name = max(candidates_recy, key=lambda t: t[0])
        return "L", name, conf

    if candidates_other:
        conf, name = max(candidates_other, key=lambda t: t[0])
        return "R", name, conf

    return None, None, 0.0


def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print(f"Connected to Arduino on {SERIAL_PORT} @ {BAUD_RATE}")

    model = YOLO(MODEL_PATH)
    names = model.names
    print(f"Loaded model: {MODEL_PATH}")

    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.2)

    # remove zoom/crop
    full_res = picam2.camera_properties["PixelArraySize"]
    picam2.set_controls({"ScalerCrop": (0, 0, full_res[0], full_res[1])})

    last_sent = 0.0
    latched = False
    missing = 0
    last_dec = None
    stable = 0

    while True:
        # Read Arduino
        while ser.in_waiting > 0:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            if "BIN1_CM" in line:
                data = parse_fill_data(line)
                if data:
                    b1_cm, b1_full, b2_cm, b2_full = data
                    print(f"\n--- BIN STATUS ---")
                    print(f"Recyclable Bin:     {b1_cm:.1f} cm | FULL: {b1_full}")
                    print(f"Non-Recyclable Bin: {b2_cm:.1f} cm | FULL: {b2_full}")
            else:
                print("[ARDUINO]", line)

        frame = picam2.capture_array()  # RGB
        h, w = frame.shape[:2]

        # ROI rect (pixel coords)
        rx1, ry1 = int(ROI_X1 * w), int(ROI_Y1 * h)
        rx2, ry2 = int(ROI_X2 * w), int(ROI_Y2 * h)
        roi = (rx1, ry1, rx2, ry2)

        # show preview + ROI
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.imshow("Sorting (q=quit) - put item in green zone", bgr)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        # YOLO
        results = model.predict(frame, imgsz=IMG_SIZE, verbose=False)
        decision, label, conf = pick_decision(results[0], names, roi)

        if decision is None:
            missing += 1
            stable = 0
            last_dec = None
            if missing >= MISSING_FRAMES_TO_RESET:
                latched = False
            time.sleep(0.01)
            continue

        missing = 0
        print(f"YOLO: {label} conf={conf:.2f} => {decision}")

        if decision == last_dec:
            stable += 1
        else:
            last_dec = decision
            stable = 1

        now = time.time()
        if (not latched) and stable >= STABLE_FRAMES_REQUIRED and (now - last_sent) >= COOLDOWN_SECONDS:
            latched = True
            last_sent = now
            ser.write(b"L" if decision == "L" else b"R")
            print(f">>> SENT {decision} ({'Recyclable' if decision=='L' else 'Non-recyclable'})")

        time.sleep(0.01)

    cv2.destroyAllWindows()
    picam2.stop()
    ser.close()


if __name__ == "__main__":
    main()
