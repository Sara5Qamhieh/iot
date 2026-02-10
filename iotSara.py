import serial
import time
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# =========================
# Serial config (Pi <-> Arduino)
# =========================
SERIAL_PORT = "/dev/ttyACM0"  # change if needed
BAUD_RATE = 115200

# =========================
# YOLO config
# =========================
MODEL_PATH = "yolov8n.pt"
CONF_THRES = 0.45
STABLE_FRAMES_REQUIRED = 4
COOLDOWN_SECONDS = 2.0

# Map detected class -> bin decision
RECYCLABLE_CLASSES = {
    "bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl",
    "book"  # tune this set (or replace with your custom model class names)
}

# =========================
# Camera config (Pi Camera Module 2 via Picamera2)
# =========================
FRAME_W = 640
FRAME_H = 480


def parse_fill_data(line: str):
    try:
        parts = line.strip().split("|")
        bin1_part = parts[0].strip()
        bin2_part = parts[1].strip()

        b1_cm = float(bin1_part.split(",")[0].split(":")[1])
        b1_full = int(bin1_part.split(",")[1].split(":")[1])

        b2_cm = float(bin2_part.split(",")[0].split(":")[1])
        b2_full = int(bin2_part.split(",")[1].split(":")[1])

        return b1_cm, b1_full, b2_cm, b2_full
    except Exception:
        return None


def decide_bin_from_detections(result, class_names) -> str | None:
    """
    Returns:
      'L' => Recyclable
      'R' => Non-recyclable
      None => no confident detection
    """
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

    cls_name = class_names.get(best_cls, str(best_cls)).strip().lower()

    if cls_name in RECYCLABLE_CLASSES:
        return "L"
    return "R"


def main():
    # ---- Serial ----
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print(f"Connected to Arduino on {SERIAL_PORT} @ {BAUD_RATE}")

    # ---- YOLO ----
    print(f"Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    class_names = model.names

    # ---- Pi Camera 2 ----
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.2)
    print("Pi Camera 2 started.")

    last_sent_time = 0.0
    stable_count = 0
    last_decision = None

    print("Running. Waiting for Arduino trigger...")

    while True:
        # 1) Read serial lines
        try:
            while ser.in_waiting > 0:
                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue

                if "BIN1_CM" in line:
                    data = parse_fill_data(line)
                    if data:
                        b1_cm, b1_full, b2_cm, b2_full = data
                        print("\n--- BIN STATUS ---")
                        print(f"Recyclable Bin:     {b1_cm:.1f} cm | FULL: {b1_full}")
                        print(f"Non-Recyclable Bin: {b2_cm:.1f} cm | FULL: {b2_full}")
                else:
                    print(f"[ARDUINO] {line}")

                # Trigger keywords (adjust to your Arduino output)
                if line.upper() in {"TRIG", "TRIGGER", "SORT", "OBJECT", "DETECT"}:
                    now = time.time()
                    if now - last_sent_time < COOLDOWN_SECONDS:
                        continue

                    print("Trigger received. Running YOLO...")
                    stable_count = 0
                    last_decision = None

                    # Collect a few frames to stabilize decision
                    for _ in range(12):
                        frame = picam2.capture_array()  # RGB888
                        # Ultralytics accepts numpy arrays; keep as RGB
                        results = model.predict(frame, imgsz=640, verbose=False)
                        decision = decide_bin_from_detections(results[0], class_names)

                        if decision is None:
                            stable_count = 0
                            last_decision = None
                            continue

                        if decision == last_decision:
                            stable_count += 1
                        else:
                            last_decision = decision
                            stable_count = 1

                        if stable_count >= STABLE_FRAMES_REQUIRED:
                            if decision == "L":
                                print("YOLO: Recyclable => sending 'L'")
                                ser.write(b"L")
                            else:
                                print("YOLO: Non-Recyclable => sending 'R'")
                                ser.write(b"R")
                            last_sent_time = time.time()
                            break

        except serial.SerialException as e:
            print(f"Serial error: {e}")
            time.sleep(1)

        # 2) Optional live preview + manual trigger
        frame = picam2.capture_array()   # RGB
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Pi Cam 2 (t=manual trigger, q=quit)", bgr)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

        if k == ord("t"):
            now = time.time()
            if now - last_sent_time >= COOLDOWN_SECONDS:
                results = model.predict(frame, imgsz=640, verbose=False)
                decision = decide_bin_from_detections(results[0], class_names)
                if decision == "L":
                    print("Manual YOLO: Recyclable => sending 'L'")
                    ser.write(b"L")
                    last_sent_time = time.time()
                elif decision == "R":
                    print("Manual YOLO: Non-Recyclable => sending 'R'")
                    ser.write(b"R")
                    last_sent_time = time.time()
                else:
                    print("Manual YOLO: No confident detection.")

        time.sleep(0.02)

    cv2.destroyAllWindows()
    picam2.stop()
    ser.close()


if __name__ == "__main__":
    main()
