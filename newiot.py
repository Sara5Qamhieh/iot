import serial
import time

# Adjust port if needed (check with: ls /dev/ttyACM*)
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # allow Arduino reset

print("Connected to Arduino")

def parse_fill_data(line):
    try:
        parts = line.strip().split("|")

        bin1_part = parts[0].strip()
        bin2_part = parts[1].strip()

        # BIN1_CM:12.3,FULL:0
        b1_cm = float(bin1_part.split(",")[0].split(":")[1])
        b1_full = int(bin1_part.split(",")[1].split(":")[1])

        b2_cm = float(bin2_part.split(",")[0].split(":")[1])
        b2_full = int(bin2_part.split(",")[1].split(":")[1])

        return b1_cm, b1_full, b2_cm, b2_full

    except:
        return None


def classify_object():
    """
    Replace this with your OpenCV classification.
    Return 'R' or 'N'
    """
    # Dummy example:
    return 'R'


while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode().strip()

        if "BIN1_CM" in line:
            data = parse_fill_data(line)
            if data:
                b1_cm, b1_full, b2_cm, b2_full = data

                print("\n--- BIN STATUS ---")
                print(f"Recyclable Bin: {b1_cm:.1f} cm | FULL: {b1_full}")
                print(f"Non-Recyclable Bin: {b2_cm:.1f} cm | FULL: {b2_full}")

    # Example trigger (replace with real vision trigger)
    # If object detected:
    if False:  # replace with your real condition
        result = classify_object()

        if result == 'R':
            print("Sending LEFT (Recyclable)")
            ser.write(b'L')
        else:
            print("Sending RIGHT (Non-Recyclable)")
            ser.write(b'R')

    time.sleep(0.1)
