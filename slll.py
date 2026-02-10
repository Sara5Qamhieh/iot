import time
import serial
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

SERIAL_PORT = "/dev/ttyUSB0"   # or /dev/ttyACM0
BAUD_RATE = 115200

MODEL_PATH = "yolo11n.pt"
IMG_SIZE = 640
CONF_THRES = 0.30

RECYCLABLE_CLASSES = {"bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl", "toothbrush"}
IGNORE_CLASSES = {"person"}

FRAME_W, FRAME_H = 1280, 960
ROI_X1, ROI_Y1 = 0.15, 0.20
ROI_X2, ROI_Y2 = 0.85, 0.95
ROI_OVERLAP_MIN = 0.15

latched = False
missing = 0
MISSING_FRAMES_TO_RESET = 12


def send_with_ack(ser, cmd: bytes, timeout=0.8, retries=5) -> bool:
    expected = b"ACK:" + cmd
    for _ in range(retries):
        try:
            ser.reset_input_buffer()
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
                if line == b"BUSY":
                    time.sleep(0.25)   # wait and retry
                    break
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
    if result.boxes is None or len(result.boxes) == 0:
        return None, None, 0.0

    best_recy = (0.0, None)
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
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)

        model = YOLO(MODEL_PATH)
        names = model.names

        picam2 = Picamera2()
        cfg = picam2.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
        picam2.configure(cfg)
        picam2.start()
        time.sleep(0.2)

        full_res = picam2.camera_properties["PixelArraySize"]
        picam2.set_controls({"ScalerCrop": (0, 0, full_res[0], full_res[1])})

        while True:
            frame = picam2.capture_array()
            h, w = frame.shape[:2]

            rx1, ry1 = int(ROI_X1 * w), int(ROI_Y1 * h)
            rx2, ry2 = int(ROI_X2 * w), int(ROI_Y2 * h)
            roi = (rx1, ry1, rx2, ry2)

            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.imshow("Sorting (q=quit)", bgr)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            results = model.predict(frame, imgsz=IMG_SIZE, verbose=False)
            decision, label, conf = pick_decision(results[0], names, roi)

            if decision is None:
                missing += 1
                if missing >= MISSING_FRAMES_TO_RESET:
                    latched = False
                continue

            missing = 0
            print(f"YOLO: {label} conf={conf:.2f} => {decision}")

            if not latched:
                latched = True

                cmd = b"L" if decision == "L" else b"R"
                ok = send_with_ack(ser, cmd)
                print(f">>> SENT {cmd.decode()} {'OK' if ok else 'FAILED'}")

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
