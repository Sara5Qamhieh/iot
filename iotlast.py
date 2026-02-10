import serial
import time
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

# ---------------- Serial ----------------
SERIAL_PORT = "/dev/ttyUSB0"   # or /dev/ttyACM0
BAUD_RATE = 115200

# ---------------- YOLO ----------------
MODEL_PATH = "yolo11n.pt"      # or "yolov8n.pt"
CONF_THRES = 0.45
COOLDOWN_SECONDS = 2.0
PROCESS_EVERY_N_FRAMES = 2     # lower = more CPU, higher = slower response

RECYCLABLE_CLASSES = {"bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl", "book"}

# ---------------- Camera ----------------
FRAME_W, FRAME_H = 1280, 960   # 4:3 helps avoid crop


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


def best_detection_decision(result, class_names):
    """Return 'L', 'R', or None."""
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best_conf = 0.0
    best_cls = None

    for b in result.boxes:
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        if conf > best_conf:
            best_conf = conf
            best_cls = cls_id

    if best_cls is None or best_conf < CONF_THRES:
        return None

    name = class_names.get(best_cls, str(best_cls)).lower().strip()
    return "L" if name in RECYCLABLE_CLASSES else "R"


def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print("Connected to Arduino.")

    model = YOLO(MODEL_PATH)
    class_names = model.names

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.2)

    # Disable crop/zoom
    full_res = picam2.camera_properties["PixelArraySize"]
    picam2.set_controls({"ScalerCrop": (0, 0, full_res[0], full_res[1])})

    print("Pi Camera started. YOLO running continuously...")

    last_sent_time = 0.0
    frame_count = 0

    while True:
        # ---- Read bin status from Arduino ----
        while ser.in_waiting > 0:
            line = ser.readline().decode(errors="ignore").strip()
            if "BIN1_CM" in line:
                data = parse_fill_data(line)
                if data:
                    b1_cm, b1_full, b2_cm, b2_full = data
                    print(f"Recyclable Bin: {b1_cm:.1f} cm | FULL: {b1_full}")
                    print(f"Non-Recyclable Bin: {b2_cm:.1f} cm | FULL: {b2_full}")
            elif line:
                print("[ARDUINO]", line)

        # ---- Capture frame ----
        frame = picam2.capture_array()  # RGB
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Pi Cam 2 (q=quit)", bgr)

        # ---- Run YOLO every N frames ----
        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            now = time.time()
            if now - last_sent_time >= COOLDOWN_SECONDS:
                results = model.predict(frame, imgsz=640, verbose=False)
                decision = best_detection_decision(results[0], class_names)

                if decision == "L":
                    print("Detected recyclable -> sending L")
                    ser.write(b"L")
                    last_sent_time = now
                elif decision == "R":
                    print("Detected non-recyclable -> sending R")
                    ser.write(b"R")
                    last_sent_time = now

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        time.sleep(0.01)

    cv2.destroyAllWindows()
    picam2.stop()
    ser.close()


if __name__ == "__main__":
    main()
