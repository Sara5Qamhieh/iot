import time
import serial
import cv2
from picamera2 import Picamera2

# --------- Serial ---------
SERIAL_PORT = "/dev/ttyACM0"   # change if needed: /dev/ttyUSB0
BAUD_RATE = 115200

# L = left 45°, R = right 45° (matches the Arduino code I gave you)
CMD_RECYCLE = b"L"
CMD_NONRECYCLE = b"R"


def parse_fill_data(line: str):
    """
    Expected line:
    BIN1_CM:12.3,FULL:0 | BIN2_CM:6.8,FULL:1
    Returns (b1_cm, b1_full, b2_cm, b2_full) or None
    """
    try:
        parts = line.strip().split("|")
        if len(parts) != 2:
            return None

        bin1_part = parts[0].strip()
        bin2_part = parts[1].strip()

        b1_cm = float(bin1_part.split(",")[0].split(":")[1])
        b1_full = int(bin1_part.split(",")[1].split(":")[1])

        b2_cm = float(bin2_part.split(",")[0].split(":")[1])
        b2_full = int(bin2_part.split(",")[1].split(":")[1])

        return b1_cm, b1_full, b2_cm, b2_full
    except:
        return None


def classify_object(frame_bgr):
    """
    TODO: Replace with your OpenCV model/classifier.
    Must return either: 'RECYCLE' or 'NONRECYCLE'

    For now, this is a placeholder that always returns RECYCLE.
    """
    # Example: show captured frame for debugging
    # cv2.imshow("Captured", frame_bgr); cv2.waitKey(500)

    return "RECYCLE"


def main():
    # Serial
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.2)
    time.sleep(2)  # Arduino reset on serial open

    # Camera v2 (Picamera2)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    print("Running.")
    print("Press 'c' to capture/classify and send servo command.")
    print("Press 'q' to quit.")

    # last known bin state (optional use)
    b1_full = 0
    b2_full = 0

    try:
        while True:
            # 1) Read from Arduino (fill levels)
            if ser.in_waiting:
                line = ser.readline().decode(errors="ignore").strip()
                if "BIN1_CM" in line and "BIN2_CM" in line:
                    data = parse_fill_data(line)
                    if data:
                        b1_cm, b1_full, b2_cm, b2_full = data
                        print(f"[RX] Bin1: {b1_cm:.1f}cm FULL:{b1_full} | Bin2: {b2_cm:.1f}cm FULL:{b2_full}")

            # 2) Live camera frame
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("PiCam v2 Preview", frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # Press 'c' to classify and sort
            if key == ord('c'):
                label = classify_object(frame_bgr)

                # Optional: block sorting if bin full (uncomment if you want)
                # if label == "RECYCLE" and b1_full == 1:
                #     print("[WARN] Recyclable bin FULL. Not sorting.")
                #     continue
                # if label == "NONRECYCLE" and b2_full == 1:
                #     print("[WARN] Non-recyclable bin FULL. Not sorting.")
                #     continue

                if label == "RECYCLE":
                    ser.write(CMD_RECYCLE)
                    print("[TX] RECYCLE -> Sent 'L' (left 45°)")
                elif label == "NONRECYCLE":
                    ser.write(CMD_NONRECYCLE)
                    print("[TX] NONRECYCLE -> Sent 'R' (right 45°)")
                else:
                    print(f"[WARN] Unknown label from classifier: {label}")

                time.sleep(0.2)  # small debounce

    finally:
        try:
            picam2.stop()
        except:
            pass
        ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
