import time
import serial
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

# =========================
# SERIAL (Pi <-> Arduino)
# =========================
SERIAL_PORT = "/dev/ttyUSB0"   # change if yours is /dev/ttyACM0
BAUD_RATE = 115200

# =========================
# YOLO
# =========================
MODEL_PATH = "yolo11n.pt"      # or "yolov8n.pt"
IMG_SIZE = 640
CONF_THRES = 0.35              # lower helps detection
STABLE_FRAMES_REQUIRED = 3     # same decision N frames in a row to trigger

# Only these classes will be considered "recyclable"
RECYCLABLE_CLASSES = {
    "bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl"
}

# We will ignore these classes for triggering (hand/person problems)
IGNORE_CLASSES = {"person"}

# =========================
# CAMERA / ROI (drop zone)
# =========================
FRAME_W, FRAME_H = 1280, 960   # 4:3 reduces cropping on Pi Cam v2

# ROI as fractions of image (tune!)
# Default: center-lower region. This helps avoid detecting your hand in upper area.
ROI_X1, ROI_Y1 = 0.30, 0.35
ROI_X2, ROI_Y2 = 0.70, 0.90

# =========================
# TRIGGER / LATCH behavior
# =========================
COOLDOWN_SECONDS = 2.0          # minimum time between triggers
MISSING_FRAMES_TO_RESET = 10    # object must disappear for N frames before next trigger


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


def point_in_roi(cx, cy, w, h):
    x1 = int(ROI_X1 * w)
    y1 = int(ROI_Y1 * h)
    x2 = int(ROI_X2 * w)
    y2 = int(ROI_Y2 * h)
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


def choose_best_object_in_roi(result, names, w, h):
    """
    Returns (class_name, confidence) for the best detection whose center is in ROI,
    ignoring IGNORE_CLASSES.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None, 0.0

    best_name = None
    best_conf = 0.0

    for b in result.boxes:
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        name = names.get(cls_id, str(cls_id)).strip().lower()

        if name in IGNORE_CLASSES:
            continue

        # box coords
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if not point_in_roi(cx, cy, w, h):
            continue

        if conf > best_conf:
            best_conf = conf
            best_name = name

    return best_name, best_conf


def decide_direction_from_name(name: str) -> str:
    # 'L' recyclable, 'R' non-recyclable
    return "L" if name in RECYCLABLE_CLASSES else "R"


def main():
    # ---- Serial ----
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print(f"Connected to Arduino on {SERIAL_PORT} @ {BAUD_RATE}")

    # ---- YOLO ----
    model = YOLO(MODEL_PATH)
    names = model.names
    print(f"Loaded model: {MODEL_PATH}")

    # ---- Camera ----
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.2)

    # Remove zoom/crop
    full_res = picam2.camera_properties["PixelArraySize"]
    picam2.set_controls({"ScalerCrop": (0, 0, full_res[0], full_res[1])})

    print("Camera started. Waiting for objects in ROI...")

    # ---- One-time trigger state ----
    last_sent_time = 0.0
    object_latched = False
    missing_frames = 0
    last_decision = None
    stable_count = 0

    while True:
        # 1) Read Arduino status (non-blocking)
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
            # ignore debug lines to reduce noise (optional)
            # else:
            #     print("[ARDUINO]", line)

        # 2) Grab frame
        frame = picam2.capture_array()  # RGB
        h, w = frame.shape[0], frame.shape[1]

        # 3) Draw ROI for debugging
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        rx1, ry1 = int(ROI_X1 * w), int(ROI_Y1 * h)
        rx2, ry2 = int(ROI_X2 * w), int(ROI_Y2 * h)
        cv2.rectangle(bgr, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.putText(bgr, "DROP ZONE", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Waste Sorting (q=quit) - Put item in GREEN box", bgr)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        # 4) Run YOLO
        results = model.predict(frame, imgsz=IMG_SIZE, verbose=False)
        name, conf = choose_best_object_in_roi(results[0], names, w, h)

        # 5) Latch logic: object must disappear before next trigger
        if name is None or conf < CONF_THRES:
            missing_frames += 1
            stable_count = 0
            last_decision = None

            if missing_frames >= MISSING_FRAMES_TO_RESET:
                object_latched = False
            time.sleep(0.01)
            continue

        # something detected in ROI
        missing_frames = 0
        decision = decide_direction_from_name(name)

        # stable decision requirement
        if decision == last_decision:
            stable_count += 1
        else:
            last_decision = decision
            stable_count = 1

        print(f"YOLO ROI sees: {name} conf={conf:.2f} -> decision {decision} (stable {stable_count}/{STABLE_FRAMES_REQUIRED})")

        now = time.time()

        # Trigger only ONCE per object appearance
        if (not object_latched) and (stable_count >= STABLE_FRAMES_REQUIRED) and ((now - last_sent_time) >= COOLDOWN_SECONDS):
            object_latched = True
            last_sent_time = now

            if decision == "L":
                print(">>> SEND L (Recyclable)")
                ser.write(b"L")
            else:
                print(">>> SEND R (Non-Recyclable)")
                ser.write(b"R")

        time.sleep(0.01)

    cv2.destroyAllWindows()
    picam2.stop()
    ser.close()


if __name__ == "__main__":
    main()
