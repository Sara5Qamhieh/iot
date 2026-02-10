import time
import serial
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

# =========================
# SERIAL
# =========================
SERIAL_PORT = "/dev/ttyUSB0"   # change if yours is /dev/ttyACM0
BAUD_RATE = 115200

# =========================
# YOLO
# =========================
MODEL_PATH = "yolo11n.pt"      # or "yolov8n.pt"
IMG_SIZE = 640
CONF_THRES = 0.30

# Map labels -> recyclable. (Add/remove as you want.)
RECYCLABLE_CLASSES = {
    "bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl",
    "toothbrush"  # you saw bottle misread as toothbrush
}
IGNORE_CLASSES = {"person"}

# =========================
# CAMERA / ROI (drop zone)
# Big ROI so partial object still counts.
# =========================
FRAME_W, FRAME_H = 1280, 960
ROI_X1, ROI_Y1 = 0.15, 0.20
ROI_X2, ROI_Y2 = 0.85, 0.95
ROI_OVERLAP_MIN = 0.15

# =========================
# ONE-SHOT trigger
# =========================
latched = False
missing = 0
MISSING_FRAMES_TO_RESET = 12


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


def send_with_ack(ser, cmd: bytes, timeout=0.7, retries=3) -> bool:
    """Send b'L' or b'R' and wait for ACK:L / ACK:R from Arduino."""
    expected = b"ACK:" + cmd
    for _ in range(retries):
        try:
            ser.reset_input_buffer()  # clear bin spam before waiting for ACK
        except Exception:
            pass

        ser.write(cmd)
        ser.flush()

        start = time.time()
        while time.time() - start < timeout:
            if ser.in_waiting:
                line = ser.readline().strip()
                if line == expected:
                    return True
        # retry
    return False


def rect_intersection_area(a, b):
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
    Return (decision 'L'/'R', label, conf) OR (None,None,0)
    - ignores 'person'
    - requires bbox overlap with ROI
    - prefers recyclable detections if present
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None, None, 0.0

    best_recy = (0.0, None)   # (conf, name)
    best_other = (0.0, None)

    for b in result.boxes:
        conf = float(b.conf[0])
        if conf < CONF_THRES:
            continue

        cls_id = int(b.cls[0])
        name = names.get(cls_id, str(cls_id)).strip().lower()

        if name in IGNORE_CLASSES:
            continue

        x1, y1, x2, y2 = b.xyxy[0].tolist()
        bbox = (x1, y1, x2, y2)

        if not bbox_overlaps_roi(bbox, roi):
            continue

        if name in RECYCLABLE_CLASSES:
            if conf > best_recy[0]:
                best_recy = (conf, name)
        else:
            if conf > best_other[0]:
                best_other = (conf, name)

    if best_recy[1] is not None:
        return "L", best_recy[1], best_recy[0]
    if best_other[1] is not None:
        return "R", best_other[1], best_other[0]

    return None, None, 0.0


def main():
    global latched, missing

    ser = None
    picam2 = None

    try:
        # Serial
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        print(f"Connected to Arduino on {SERIAL_PORT} @ {BAUD_RATE}")

        # YOLO
        model = YOLO(MODEL_PATH)
        names = model.names
        print(f"Loaded model: {MODEL_PATH}")

        # Camera
        picam2 = Picamera2()
        cfg = picam2.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
        picam2.configure(cfg)
        picam2.start()
        time.sleep(0.2)

        # Remove crop/zoom
        full_res = picam2.camera_properties["PixelArraySize"]
        picam2.set_controls({"ScalerCrop": (0, 0, full_res[0], full_res[1])})

        print("Camera started. Put the item in the green box.")

        while True:
            # Read Arduino messages (bin status + ACK + debug)
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

            rx1, ry1 = int(ROI_X1 * w), int(ROI_Y1 * h)
            rx2, ry2 = int(ROI_X2 * w), int(ROI_Y2 * h)
            roi = (rx1, ry1, rx2, ry2)

            # Preview (remove if you want headless)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.imshow("Sorting (q=quit)", bgr)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            # YOLO
            results = model.predict(frame, imgsz=IMG_SIZE, verbose=False)
            decision, label, conf = pick_decision(results[0], names, roi)

            # Nothing detected in ROI -> unlock after a while
            if decision is None:
                missing += 1
                if missing >= MISSING_FRAMES_TO_RESET:
                    latched = False
                continue

            # Something detected
            missing = 0
            print(f"YOLO: {label} conf={conf:.2f} => {decision}")

            # SEND ONCE ONLY
            if not latched:
                latched = True

                # If you want recyclable to go RIGHT, flip here:
                # cmd = b"R" if decision == "L" else b"L"
                cmd = b"L" if decision == "L" else b"R"

                ok = send_with_ack(ser, cmd)
                print(f">>> SENT {cmd.decode()} {'OK' if ok else 'FAILED'}")

        cv2.destroyAllWindows()

    finally:
        try:
            cv2.destroyAllWindows()
        except:
            pass
        try:
            if picam2 is not None:
                picam2.stop()
                picam2.close()
        except:
            pass
        try:
            if ser is not None:
                ser.close()
        except:
            pass


if __name__ == "__main__":
    main()
