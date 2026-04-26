import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
import cv2
from matplotlib import cm

from model import ShapeNet

# ---------------- Config ----------------
MODEL_PATH = "best_combined.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPECTED_COLUMNS = 218
COLUMNS = np.concatenate([np.arange(9 + 7 * i, 15 + 7 * i) for i in range(30)])
WIDTH, HEIGHT = 960, 720
DEFAULT_FPS = 30.0
OUTPUT_FPS = 30.0
KALMAN_ENABLED = True
KALMAN_Q = 0.05   # process noise variance (lower = smoother, slower to react)
KALMAN_R = 1.0    # measurement noise variance

# ---------------- Model -----------------
model = ShapeNet(
    pd_in_features=180, latent_dim=512, num_points=4096,
    tnet1=False, tnet2=False
).to(DEVICE).eval()
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.pd2latent.load_state_dict(ckpt["pd2latent_state_dict"])
model.decoder.load_state_dict(ckpt["decoder_state_dict"])
torch.backends.cudnn.benchmark = True


def infer(optical_vals: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(optical_vals).float().unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        pred = model(x)
    if pred.dim() == 3:
        pred = pred[0]
    return pred.cpu().numpy().astype(np.float32)


def fps_from_timestamps(timestamps: np.ndarray) -> float:
    ts = timestamps.astype(np.float64)
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return DEFAULT_FPS
    median_dt = np.median(diffs)
    # If median_dt > 1000, assume microseconds; otherwise milliseconds
    if median_dt > 1000:
        fps = 1_000_000.0 / median_dt
    else:
        fps = 1_000.0 / median_dt
    return float(np.clip(fps, 1.0, 240.0))


def kalman_smooth(optical_seq: np.ndarray, q: float = KALMAN_Q, r: float = KALMAN_R) -> np.ndarray:
    """Causal Kalman filter over a (T, D) optical sequence, applied per-feature.

    Assumes a constant-velocity model (identity transition) with diagonal
    covariance, so the per-feature updates are fully independent and vectorized.
    """
    T, D = optical_seq.shape
    smoothed = np.empty_like(optical_seq)

    x = optical_seq[0].copy()          # state estimate
    P = np.full(D, r, dtype=np.float64) # initial covariance — start uncertain

    Q = np.full(D, q, dtype=np.float64)
    R = np.full(D, r, dtype=np.float64)

    for t in range(T):
        # Predict (identity transition)
        P_pred = P + Q

        # Update
        K = P_pred / (P_pred + R)      # Kalman gain, shape (D,)
        x = x + K * (optical_seq[t] - x)
        P = (1.0 - K) * P_pred

        smoothed[t] = x

    return smoothed


def main():
    parser = argparse.ArgumentParser(
        description="Render optical npy frames as a real-time point-cloud video"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=str(Path(__file__).parent.parent / "Data" / "Test_1"),
        help="Directory containing *_optical.npy files (default: Data/Test_1)",
    )
    parser.add_argument("--output", "-o", default=None,
                        help="Output video path (default: <directory>/output.mp4)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override FPS; by default derived from timestamps")
    args = parser.parse_args()

    data_dir = Path(args.directory)
    if not data_dir.exists():
        print(f"Error: directory '{data_dir}' does not exist")
        sys.exit(1)

    npy_files = sorted(data_dir.glob("*_optical.npy"))
    if not npy_files:
        print(f"No *_optical.npy files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(npy_files)} files in {data_dir}")

    # -------- pass 1: collect valid timestamps and optical data (no inference) --------
    valid_frames = []  # list of (timestamp, optical)
    for i, f in enumerate(npy_files):
        arr = np.load(f).astype(np.float32)
        if arr.shape[0] != EXPECTED_COLUMNS:
            print(f"  skip {f.name}: expected {EXPECTED_COLUMNS} values, got {arr.shape[0]}")
            continue

        timestamp = arr[0]
        optical = arr[COLUMNS]

        if optical.shape[0] != 180 or not np.isfinite(optical).all():
            print(f"  skip {f.name}: invalid optical data")
            continue

        valid_frames.append((timestamp, optical))

    n_valid = len(valid_frames)
    if n_valid == 0:
        print("No valid frames found")
        sys.exit(1)

    print(f"Valid frames: {n_valid}")

    # -------- Kalman filter optical sequences before any frame dropping --------
    if KALMAN_ENABLED:
        optical_seq = np.stack([opt for _, opt in valid_frames], axis=0)  # (T, 180)
        optical_seq = kalman_smooth(optical_seq)
        valid_frames = [(ts, optical_seq[i]) for i, (ts, _) in enumerate(valid_frames)]
        print(f"Kalman filtering done (Q={KALMAN_Q}, R={KALMAN_R})")
    else:
        print("Kalman filtering disabled")

    # -------- derive source FPS and select frame indices for OUTPUT_FPS output --------
    source_fps = args.fps
    if source_fps is None:
        timestamps_arr = np.array([ts for ts, _ in valid_frames])
        source_fps = fps_from_timestamps(timestamps_arr)
        print(f"Derived source FPS from timestamps: {source_fps:.2f}")

    step = source_fps / OUTPUT_FPS
    raw_indices = np.arange(0, n_valid, step)
    selected_indices = np.unique(np.clip(np.round(raw_indices).astype(int), 0, n_valid - 1))
    print(f"Subsampling {n_valid} frames at step {step:.2f} -> {len(selected_indices)} frames @ {OUTPUT_FPS:.0f} FPS")

    # -------- pass 2: inference on selected frames only --------
    all_pts = []
    for j, idx in enumerate(selected_indices):
        _, optical = valid_frames[idx]
        all_pts.append(infer(optical))

        if (j + 1) % 20 == 0 or (j + 1) == len(selected_indices):
            print(f"  inference {j+1}/{len(selected_indices)}")

    n_frames = len(all_pts)
    print(f"Inference done: {n_frames} frames")

    fps = OUTPUT_FPS

    # -------- Visualizer (hidden window — works on Windows where EGL is unavailable) --------
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=WIDTH, height=HEIGHT, visible=False)

    render_opt = vis.get_render_option()
    render_opt.point_size = 3.0
    render_opt.background_color = np.array([0.0, 0.0, 0.0])

    # -------- video writer --------
    out_path = args.output or str(data_dir / "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        print(f"Error: could not open video writer for '{out_path}'")
        vis.destroy_window()
        sys.exit(1)

    # -------- render loop --------
    viridis = cm.get_cmap("viridis")
    zmin, zmax = 0.0, 20.0
    alpha = 0.15
    pcd = o3d.geometry.PointCloud()
    saved_cam = None

    for i, pts in enumerate(all_pts):
        z = -pts[:, 2]
        zmin = (1.0 - alpha) * zmin + alpha * float(np.min(z))
        zmax = (1.0 - alpha) * zmax + alpha * float(np.max(z))
        if zmax <= zmin:
            zmax = zmin + 1e-6
        nz = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)
        col = viridis(nz)[:, :3]

        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64))

        if i == 0:
            vis.add_geometry(pcd, reset_bounding_box=True)
            ctr = vis.get_view_control()
            ctr.set_lookat(pts.mean(axis=0).tolist())
            ctr.set_front([0.0, 0.0, -1.0])
            ctr.set_up([0.0, -1.0, 0.0])
            ctr.set_zoom(0.8)
            saved_cam = ctr.convert_to_pinhole_camera_parameters()
        else:
            if saved_cam is not None:
                vis.get_view_control().convert_from_pinhole_camera_parameters(
                    saved_cam, allow_arbitrary=True
                )
            vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

        # capture_screen_float_buffer returns (H, W, 3) float32 in [0, 1]
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame_uint8 = (frame * 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))

        if (i + 1) % 20 == 0 or (i + 1) == n_frames:
            print(f"  rendered {i+1}/{n_frames}")

    writer.release()
    vis.destroy_window()
    print(f"Saved: {out_path}  ({n_frames} frames @ {fps:.0f} FPS, subsampled from {source_fps:.1f} FPS source)")


if __name__ == "__main__":
    main()
