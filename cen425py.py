import serial
import time
from RPLCD.i2c import CharLCD

# -------- SERIAL --------
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

# -------- LCD SETUP --------
# Change address to 0x3f if needed
lcd = CharLCD(
    i2c_expander='PCF8574',
    address=0x27,
    port=1,
    cols=16,
    rows=2,
    dotsize=8
)

lcd.clear()
lcd.write_string("Smart Waste Bin")
time.sleep(2)
lcd.clear()


def parse_fill_data(line):
    try:
        parts = line.strip().split("|")

        bin1_part = parts[0].strip()
        bin2_part = parts[1].strip()

        b1_cm = float(bin1_part.split(",")[0].split(":")[1])
        b1_full = int(bin1_part.split(",")[1].split(":")[1])

        b2_cm = float(bin2_part.split(",")[0].split(":")[1])
        b2_full = int(bin2_part.split(",")[1].split(":")[1])

        return b1_cm, b1_full, b2_cm, b2_full
    except:
        return None


def update_lcd(b1_cm, b1_full, b2_cm, b2_full):
    lcd.clear()

    line1 = f"B1:{b1_cm:4.1f}cm"
    if b1_full:
        line1 += " FULL"
    else:
        line1 += " OK  "

    line2 = f"B2:{b2_cm:4.1f}cm"
    if b2_full:
        line2 += " FULL"
    else:
        line2 += " OK  "

    lcd.cursor_pos = (0, 0)
    lcd.write_string(line1[:16])

    lcd.cursor_pos = (1, 0)
    lcd.write_string(line2[:16])


while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode().strip()

        if "BIN1_CM" in line:
            data = parse_fill_data(line)
            if data:
                b1_cm, b1_full, b2_cm, b2_full = data

                print(f"Bin1: {b1_cm:.1f}cm FULL:{b1_full}")
                print(f"Bin2: {b2_cm:.1f}cm FULL:{b2_full}")

                update_lcd(b1_cm, b1_full, b2_cm, b2_full)

    time.sleep(0.1)
