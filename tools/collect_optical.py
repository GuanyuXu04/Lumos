from pathlib import Path
import time
import re

import numpy as np
import serial

SENSOR_WIDTH = 275
SENSOR_HEIGHT = 275
SESSION_NAME = "Test_1"
SESSION_DIR = Path(__file__).parent.parent / "Data" / SESSION_NAME
NUM_LED = 30
NUM_PD = 6

BAUD_RATE = 2000000
SERIAL_PORT = "COM6"
EXPECTED_VALS = 1 + (NUM_LED + 1) * (NUM_PD + 1)

_ID_REGEX = re.compile(r"frame_(\d+)_.*\.npy$", re.IGNORECASE)

def _next_start_id(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    max_id = -1
    for p in directory.glob("*.npy"):
        m = _ID_REGEX.search(p.name)
        if m:
            try:
                n = int(m.group(1))
                if n > max_id:
                    max_id = n
            except Exception:
                pass
    return max_id + 1


def main():
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    start_id = _next_start_id(SESSION_DIR)
    print(f"[id] Start global frame id = {start_id} (scan {SESSION_NAME}/)")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.0)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Failed to open serial {SERIAL_PORT}: {e}")
        return

    ser_buf = b""
    count = 0
    print("Logging... (Ctrl+C to stop)")

    try:
        while True:
            waiting = ser.in_waiting
            if not waiting:
                time.sleep(0.0005)
                continue

            ser_buf += ser.read(waiting)
            last_nl = ser_buf.rfind(b"\n")
            if last_nl == -1:
                continue

            chunk = ser_buf[:last_nl]
            ser_buf = ser_buf[last_nl + 1:]
            lines = chunk.split(b"\n")

            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if not line:
                    continue
                try:
                    txt = line.decode("utf-8", errors="ignore").strip("\r")
                    arr = np.fromstring(txt, dtype=np.int32, sep=",")
                    if arr.size == EXPECTED_VALS:
                        global_id = start_id + count
                        np.save(SESSION_DIR / f"frame_{global_id:06d}_optical.npy", arr)
                        count += 1
                        print(arr)
                except Exception:
                    pass
                break

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    finally:
        try:
            ser.close()
        except Exception:
            pass

    print(f"Saved {count} optical frames into {SESSION_DIR} (starting from id {start_id})")


if __name__ == "__main__":
    main()
