"""
Read RealSense depth + Optical Sensor (COM9) in one loop (no threads).

D435 settings:
- 848x480 @ 30 FPS, z16
- Emitter enabled; laser_power=360
- Auto Exposure OFF; exposure=1000 us; gain=16
- Global Time Enabled
- Try depth_units = 0.0005 (if supported)

== ROI ==
Use a fixed 230x230 square centered in the depth frame.
Show live view with square; press Enter to start logging.

== Naming ==
All per-frame .npy files go to ./Combined_Data/
At program start, scan Combined_Data and start numbering from (max_id + 1).
Saved per frame:
- Combined_Data/frame_{id:06d}_depth.npy
- Combined_Data/frame_{id:06d}_optical.npy
"""

from pathlib import Path
from datetime import datetime
import time
import re

import numpy as np
import cv2
import serial
import pyrealsense2 as rs

# -------------------- User settings --------------------
SERIAL_PORT   = "COM6"
BAUD_RATE     = 2_000_000
SER_TIMEOUT   = 0.0             # non-blocking
EXPECTED_VALS = 218

SESSION_DIR   = Path("Combined_Data")  # per-run meta / logs
COMBINED_DIR  = SESSION_DIR
SHOW_WINDOW   = "RealSense Depth (center square ROI / press Enter to start)"

# Depth stream format
DEPTH_W, DEPTH_H, DEPTH_FPS = 848, 480, 30

MAX_PAIRS = 3000000

# ---------------- Depth configuration (match screenshot) ----------------
USE_HIGH_DENSITY_PRESET = False
ENABLE_EMITTER = True
LASER_POWER_ABS = 360.0
SET_CONFIDENCE_THRESHOLD = False
CONFIDENCE_THRESHOLD = 3

USE_MANUAL_EXPOSURE = True
MANUAL_EXPOSURE_US  = 1000.0
MANUAL_GAIN         = 16.0

ENABLE_GLOBAL_TIME = True
TRY_SET_DEPTH_UNITS = True
DEPTH_UNITS_VALUE = 0.0005

# ---------------- Optional post-processing ----------------
USE_DECIMATION = False
DECIMATION_MAG = 1
USE_SPATIAL    = False
SPATIAL_MAG = 1
SPATIAL_SMOOTH_ALPHA = 0.25
SPATIAL_SMOOTH_DELTA = 10
SPATIAL_HOLES_FILL = 2
USE_TEMPORAL   = True
USE_HOLE_FILLING = True
HOLE_FILLING_MODE = 0
# ---------------------------------------------------------------------

# ---------- Preview performance knobs ----------
PREVIEW_EVERY   = 10
PREVIEW_DOWNSCALE = 2
# ------------------------------------------------------


def build_labels() -> list[str]:
    labels = ["time_ms"]
    for led in range(31):
        for pd in range(7):
            labels.append(f"L{led}P{pd}")
    assert len(labels) == EXPECTED_VALS
    return labels


def _safe_set(sensor: rs.sensor, opt: rs.option, value: float, label: str):
    try:
        if sensor.supports(opt):
            rng = sensor.get_option_range(opt)
            val = max(rng.min, min(rng.max, float(value)))
            sensor.set_option(opt, val)
            print(f"[cfg] {label} = {val} (range {rng.min}-{rng.max})")
        else:
            print(f"[cfg] {label} not supported on this sensor")
    except Exception as e:
        print(f"[cfg] Failed to set {label}: {e}")


def start_depth_pipeline(w=DEPTH_W, h=DEPTH_H, fps=DEPTH_FPS) -> tuple[rs.pipeline, rs.pipeline_profile]:
    pipeline = rs.pipeline()
    config = rs.config()
    try:
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        profile = pipeline.start(config)
        print(f"[ok] Depth started: {w}x{h}@{fps} z16")
    except Exception as e:
        print(f"[warn] Requested {w}x{h}@{fps} failed: {e}")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth)
        profile = pipeline.start(config)
        sp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = sp.get_intrinsics()
        print(f"[ok] Fallback depth started: {intr.width}x{intr.height}@{sp.fps()} z16")

    # ---- Configure Stereo Module ----
    try:
        device = profile.get_device()
        stereo = None
        for s in device.query_sensors():
            if s.get_info(rs.camera_info.name) == "Stereo Module":
                stereo = s; break

        if stereo:
            if USE_HIGH_DENSITY_PRESET and stereo.supports(rs.option.visual_preset):
                stereo.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_density)
                print("[cfg] visual_preset = high_density")

            if ENABLE_EMITTER and stereo.supports(rs.option.emitter_enabled):
                stereo.set_option(rs.option.emitter_enabled, 1)
                print("[cfg] emitter_enabled = 1")

            _safe_set(stereo, rs.option.laser_power, LASER_POWER_ABS, "laser_power")

            if ENABLE_GLOBAL_TIME and stereo.supports(rs.option.global_time_enabled):
                try:
                    stereo.set_option(rs.option.global_time_enabled, 1)
                    print("[cfg] global_time_enabled = 1")
                except Exception as e:
                    print(f"[cfg] global_time_enabled not set: {e}")

            if stereo.supports(rs.option.enable_auto_exposure):
                stereo.set_option(rs.option.enable_auto_exposure, 0 if USE_MANUAL_EXPOSURE else 1)
                print(f"[cfg] auto_exposure = {0 if USE_MANUAL_EXPOSURE else 1}")
            if USE_MANUAL_EXPOSURE:
                _safe_set(stereo, rs.option.exposure, MANUAL_EXPOSURE_US, "exposure_us")
                _safe_set(stereo, rs.option.gain, MANUAL_GAIN, "gain")

            if SET_CONFIDENCE_THRESHOLD:
                _safe_set(stereo, rs.option.confidence_threshold, CONFIDENCE_THRESHOLD, "confidence_threshold")

            if TRY_SET_DEPTH_UNITS and stereo.supports(rs.option.depth_units):
                try:
                    stereo.set_option(rs.option.depth_units, float(DEPTH_UNITS_VALUE))
                    print(f"[cfg] depth_units = {DEPTH_UNITS_VALUE}")
                except Exception as e:
                    print(f"[cfg] depth_units not set: {e}")

    except Exception as e:
        print(f"[warn] Stereo config skipped: {e}")

    return pipeline, profile


def create_filters():
    flts = []
    if USE_DECIMATION and DECIMATION_MAG != 1:
        dec = rs.decimation_filter()
        dec.set_option(rs.option.filter_magnitude, float(DECIMATION_MAG))
        flts.append(dec)
    if USE_SPATIAL:
        spat = rs.spatial_filter()
        spat.set_option(rs.option.filter_magnitude, float(SPATIAL_MAG))
        spat.set_option(rs.option.filter_smooth_alpha, float(SPATIAL_SMOOTH_ALPHA))
        spat.set_option(rs.option.filter_smooth_delta, float(SPATIAL_SMOOTH_DELTA))
        spat.set_option(rs.option.holes_fill, float(SPATIAL_HOLES_FILL))
        flts.append(spat)
    if USE_TEMPORAL:
        flts.append(rs.temporal_filter())
    if USE_HOLE_FILLING:
        flts.append(rs.hole_filling_filter(HOLE_FILLING_MODE))
    return flts


def process_depth_frame(depth_frame, filters):
    f = depth_frame
    for flt in filters:
        f = flt.process(f)
    return f.as_depth_frame()


def select_center_square_live(pipeline: rs.pipeline, colorizer: rs.colorizer, filters,
                              w_box: int = 230, h_box: int = 230):
    """Live view with fixed central square; press Enter to confirm."""
    cv2.namedWindow(SHOW_WINDOW, cv2.WINDOW_NORMAL)

    x0 = y0 = None
    roi_mask_crop = None
    poly_pts = None

    while True:
        frames = pipeline.wait_for_frames()
        df = frames.get_depth_frame()
        if not df:
            continue
        if filters:
            df = process_depth_frame(df, filters)

        disp = np.asanyarray(colorizer.colorize(df).get_data())
        H, W = disp.shape[:2]

        if x0 is None:
            x0 = (W - w_box) // 2
            y0 = (H - h_box) // 2

            poly_pts = np.array([[x0, y0],
                                 [x0 + w_box - 1, y0],
                                 [x0 + w_box - 1, y0 + h_box - 1],
                                 [x0, y0 + h_box - 1]], dtype=np.int32)

            full_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(full_mask, [poly_pts], color=1)
            roi_mask_crop = full_mask[y0:y0 + h_box, x0:x0 + w_box].copy()

        cv2.rectangle(disp, (x0, y0), (x0 + w_box - 1, y0 + h_box - 1), (0, 255, 0), 2)
        cv2.putText(disp, "Align inside square, press Enter to start (q/Esc to quit)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(SHOW_WINDOW, disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            raise KeyboardInterrupt("User quit before start.")
        if key == 13:
            break

    return poly_pts, (int(x0), int(y0)), roi_mask_crop


# ---------- Global ID helpers ----------
_ID_REGEX = re.compile(r"frame_(\d+)_.*\.npy$", re.IGNORECASE)

def _scan_max_id(directory: Path) -> int:
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
    return max_id

def _next_start_id(directory: Path) -> int:
    return _scan_max_id(directory) + 1


def main():
    # session dir for meta
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SESSION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # combined dir for global frames
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    start_id = _next_start_id(COMBINED_DIR)
    print(f"[id] Start global frame id = {start_id} (scan Combined_Data/)")

    # ---- Restore session config from existing meta (if any) ----
    # The session is identified by SESSION_DIR. Once meta.npz is written, every
    # subsequent run in the same session must capture with the SAME camera
    # config so frames stay self-consistent. If a run silently reset the
    # hardware to different depth_units / exposure / ROI, downstream code
    # interprets old frames with the wrong scale (this caused the i<161489
    # patches in lumos/data.py).
    meta_path = out_dir / "meta.npz"
    existing_meta = None
    if meta_path.exists():
        existing_meta = np.load(meta_path, allow_pickle=True)
        global ENABLE_EMITTER, LASER_POWER_ABS, USE_MANUAL_EXPOSURE
        global MANUAL_EXPOSURE_US, MANUAL_GAIN, ENABLE_GLOBAL_TIME, DEPTH_UNITS_VALUE
        ENABLE_EMITTER      = bool(int(existing_meta["cfg_emitter"][0]))
        LASER_POWER_ABS     = float(existing_meta["cfg_laser_power"][0])
        USE_MANUAL_EXPOSURE = bool(int(existing_meta["cfg_manual_exposure"][0]))
        MANUAL_EXPOSURE_US  = float(existing_meta["cfg_exposure_us"][0])
        MANUAL_GAIN         = float(existing_meta["cfg_gain"][0])
        ENABLE_GLOBAL_TIME  = bool(int(existing_meta["cfg_global_time_enabled"][0]))
        DEPTH_UNITS_VALUE   = float(existing_meta["cfg_depth_units_target"][0])
        print(f"[meta] Restoring camera config from {meta_path}")
    elif start_id > 0:
        print(f"[warn] {start_id} frames already in {COMBINED_DIR} but no meta.npz; "
              "those frames will be re-keyed to a fresh meta")

    # ---- Start RealSense ----
    pipeline, profile = start_depth_pipeline()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    colorizer = rs.colorizer()

    depth_sp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = depth_sp.get_intrinsics()
    W_full, H_full = intr.width, intr.height
    fx, fy, ppx, ppy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)
    dist_coeffs = np.array(intr.coeffs, dtype=np.float64)
    depth_model = int(intr.model)

    K_full = np.array([[fx, 0.0, ppx],
                       [0.0, fy, ppy],
                       [0.0, 0.0, 1.0]], dtype=np.float64)

    filters = create_filters()

    if existing_meta is not None:
        # Validate that the camera actually accepted the requested config; if it
        # didn't, frames captured now would be inconsistent with prior frames
        # in this session. Refuse rather than silently corrupt the dataset.
        saved_scale = float(existing_meta["depth_scale_m_per_unit"])
        if abs(depth_scale - saved_scale) > 1e-9:
            pipeline.stop()
            raise RuntimeError(
                f"Depth scale mismatch: camera reports {depth_scale}, session meta "
                f"requires {saved_scale}. The camera likely did not accept "
                f"depth_units={DEPTH_UNITS_VALUE}. Aborting to keep "
                f"{COMBINED_DIR} consistent."
            )

        saved_size = existing_meta["depth_size_full_wh"]
        if int(saved_size[0]) != W_full or int(saved_size[1]) != H_full:
            pipeline.stop()
            raise RuntimeError(
                f"Depth stream size mismatch: camera {W_full}x{H_full}, session "
                f"meta {int(saved_size[0])}x{int(saved_size[1])}."
            )

        # Reuse the saved ROI verbatim — no live selector.
        x0, y0 = (int(v) for v in existing_meta["roi_origin_xy"])
        roi_mask = existing_meta["roi_mask"].astype(np.uint8)
        poly_pts = existing_meta["roi_polygon_xy"].astype(np.int32)
        h_roi, w_roi = roi_mask.shape
        K_roi = existing_meta["depth_K_roi"].astype(np.float64)
        print(f"[meta] Reusing saved ROI origin=({x0},{y0}) size={w_roi}x{h_roi}")
    else:
        # ---- First run in this session: pick ROI live and persist meta ----
        try:
            poly_pts, (x0, y0), roi_mask = select_center_square_live(
                pipeline, colorizer, filters, w_box=230, h_box=230
            )
        except KeyboardInterrupt:
            pipeline.stop(); cv2.destroyAllWindows()
            return

        h_roi, w_roi = roi_mask.shape
        K_roi = K_full.copy()
        K_roi[0, 2] -= float(x0)
        K_roi[1, 2] -= float(y0)

        meta = {
            "roi_origin_xy": np.array([x0, y0], dtype=np.int32),
            "roi_mask": roi_mask.astype(np.uint8),
            "roi_polygon_xy": poly_pts.astype(np.int32),
            "depth_scale_m_per_unit": depth_scale,
            "depth_K_full": K_full,
            "depth_size_full_wh": np.array([W_full, H_full], dtype=np.int32),
            "depth_K_roi": K_roi,
            "depth_size_roi_wh": np.array([w_roi, h_roi], dtype=np.int32),
            "depth_dist_coeffs": dist_coeffs,
            "depth_dist_model": np.array([depth_model], dtype=np.int32),
            "labels": np.array(build_labels(), dtype=object),

            # key runtime cfg
            "cfg_emitter": np.array([int(ENABLE_EMITTER)], dtype=np.int32),
            "cfg_laser_power": np.array([LASER_POWER_ABS], dtype=np.float32),
            "cfg_manual_exposure": np.array([int(USE_MANUAL_EXPOSURE)], dtype=np.int32),
            "cfg_exposure_us": np.array([MANUAL_EXPOSURE_US], dtype=np.float32),
            "cfg_gain": np.array([MANUAL_GAIN], dtype=np.float32),
            "cfg_global_time_enabled": np.array([int(ENABLE_GLOBAL_TIME)], dtype=np.int32),
            "cfg_depth_units_target": np.array([DEPTH_UNITS_VALUE], dtype=np.float32),
        }
        np.savez_compressed(meta_path, **meta)

    # ---- Serial ----
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SER_TIMEOUT)
        ser.reset_input_buffer()
    except Exception as e:
        pipeline.stop(); cv2.destroyAllWindows()
        print(f"Failed to open serial {SERIAL_PORT}: {e}")
        return

    ser_buf = b""
    pending_optical = None
    pairs = 0
    preview_counter = 0
    print("Logging...")

    try:
        while True:
            if MAX_PAIRS and pairs >= MAX_PAIRS:
                break

            frames = pipeline.wait_for_frames()
            d = frames.get_depth_frame()
            if not d:
                continue
            if filters:
                d = process_depth_frame(d, filters)
            arr = np.asanyarray(d.get_data())
            crop = arr[y0:y0 + h_roi, x0:x0 + w_roi].copy()
            crop[roi_mask == 0] = 0

            waiting = ser.in_waiting if ser else 0
            if waiting:
                ser_buf += ser.read(waiting)
                last_nl = ser_buf.rfind(b"\n")
                if last_nl != -1:
                    chunk = ser_buf[:last_nl]
                    ser_buf = ser_buf[last_nl + 1:]
                    lines = chunk.split(b"\n")
                    for i in range(len(lines)-1, -1, -1):
                        line = lines[i].strip()
                        if line:
                            try:
                                txt = line.decode("utf-8", errors="ignore").strip("\r")
                                arr_i = np.fromstring(txt, dtype=np.int32, sep=",")
                                if arr_i.size == EXPECTED_VALS:
                                    pending_optical = (arr_i, time.time_ns())
                                    print(arr_i)
                            except Exception:
                                pass
                            break

            if pending_optical is not None:
                ovals, _o_host_ns = pending_optical
                # compute global id
                global_id = start_id + pairs

                # save immediately into Combined_Data
                np.save(COMBINED_DIR / f"frame_{global_id:06d}_depth.npy", crop.astype(np.uint16))
                np.save(COMBINED_DIR / f"frame_{global_id:06d}_optical.npy", ovals.astype(np.int32))

                pairs += 1
                pending_optical = None

                preview_counter += 1
                if preview_counter % PREVIEW_EVERY == 0:
                    disp = np.asanyarray(colorizer.colorize(d).get_data())
                    cv2.polylines(disp, [poly_pts], True, (0, 255, 0), 2)
                    if PREVIEW_DOWNSCALE > 1:
                        disp = cv2.resize(disp, (disp.shape[1] // PREVIEW_DOWNSCALE,
                                                 disp.shape[0] // PREVIEW_DOWNSCALE))
                    cv2.imshow(SHOW_WINDOW, disp)

            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    finally:
        try: ser.close()
        except: pass
        try: pipeline.stop()
        except: pass
        cv2.destroyAllWindows()

    print(f"Saved {pairs} pairs into {COMBINED_DIR} (starting from id {start_id})")


if __name__ == "__main__":
    try: cv2.setUseOptimized(True)
    except Exception: pass
    main()
