import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pandas as pd  # NEW

from lumos.model import ShapeNet
from lumos.data import _load_meta, _depth_to_points_mm_fixedgrid

# ---------------- User config ----------------
MODEL_PATH = "checkpoints/best_combined_500.pth"
DATA_PATH  = "Data/Combined_Data"
STRIDE = 3

TEST_START_IDX = 10000
TEST_FRAMES    = 10000

OUT_MP4   = "gt_vs_pr.mp4"
OUT_FPS   = 30
POINT_SIZE = 1
DPI = 140
FIGSIZE = (12, 6)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Optical CSV config (NEW) ----------------
OPTICAL_CSV = "test_poke_kalman.csv"
CSV_TO_DEPTH_OFFSET = 242689  # csv_id + 242689 = depth frame id
OPTICAL_COLS = [f"L{i}P5" for i in range(1, 31)]  # 30 dims: L1P5~L30P5

# ---------------- Load optical CSV once (NEW) ----------------
df_opt = pd.read_csv(OPTICAL_CSV)

# basic checks
if "frame_id" not in df_opt.columns:
    raise ValueError(f"{OPTICAL_CSV} missing required column: frame_id")
missing_cols = [c for c in OPTICAL_COLS if c not in df_opt.columns]
if missing_cols:
    raise ValueError(f"{OPTICAL_CSV} missing required optical columns: {missing_cols}")

# Use frame_id as index for fast lookup
df_opt = df_opt.set_index("frame_id").sort_index()

# ---------------- Model ----------------
model = ShapeNet(
    pd_in_features=30,   # <-- ensure 30
    latent_dim=512,
    num_points=4096,
    tnet1=False, tnet2=False
).to(DEVICE).eval()

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.pd2latent.load_state_dict(ckpt["pd2latent_state_dict"])
model.decoder.load_state_dict(ckpt["decoder_state_dict"])
torch.backends.cudnn.benchmark = True

# ---------------- Meta (same as dataset uses) ----------------
roi_mask, scale_m, K = _load_meta(os.path.join(DATA_PATH, "meta.npz"))

def npy_path(fid: int, kind: str) -> str:
    return os.path.join(DATA_PATH, f"frame_{fid:06d}_{kind}.npy")

def load_gt_points_and_optical(fid: int):
    # depth: 2D array (H,W), usually uint16
    depth = np.load(npy_path(fid, "depth"), mmap_mode="r")
    pts = _depth_to_points_mm_fixedgrid(depth, roi_mask, K, scale_m, stride=STRIDE)

    # match data.py special-case scaling (original code checks i<161489)
    if fid < 161489 or fid > 242688:
        pts *= 2.0

    # -------- optical from CSV (NEW) --------
    csv_id = fid - CSV_TO_DEPTH_OFFSET
    if csv_id not in df_opt.index:
        raise KeyError(
            f"optical csv missing frame_id={csv_id} (depth fid={fid}, offset={CSV_TO_DEPTH_OFFSET})"
        )

    optical = df_opt.loc[csv_id, OPTICAL_COLS].to_numpy(dtype=np.float32)
    # ensure shape (30,)
    optical = np.asarray(optical, dtype=np.float32).reshape(-1)
    if optical.shape[0] != 30:
        raise ValueError(f"Expected 30-d optical, got {optical.shape} at csv_id={csv_id}")

    return pts.astype(np.float32), optical

def render_figure_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)

def set_axes_common(ax, xlim, ylim, zlim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

# ---------------- Pre-pass: global limits (avoid jitter) ----------------
x_min, y_min, z_min = np.inf, np.inf, np.inf
x_max, y_max, z_max = -np.inf, -np.inf, -np.inf

for i in range(TEST_FRAMES):
    fid = TEST_START_IDX + i
    gt_pts, _ = load_gt_points_and_optical(fid)

    x_min = min(x_min, float(gt_pts[:, 0].min()))
    x_max = max(x_max, float(gt_pts[:, 0].max()))
    y_min = min(y_min, float(gt_pts[:, 1].min()))
    y_max = max(y_max, float(gt_pts[:, 1].max()))
    z_min = min(z_min, float(gt_pts[:, 2].min()))
    z_max = max(z_max, float(gt_pts[:, 2].max()))

xlim = (x_min, x_max)
ylim = (y_min, y_max)
zlim = (-z_max, -z_min)     # evaluate.py 里画的是 -z
vmin, vmax = z_min, z_max   # 颜色用原始 z

print("[Global limits]")
print("xlim:", xlim, "ylim:", ylim, "zlim(display):", zlim, "color(z) range:", (vmin, vmax))

# ---------------- Video writer ----------------
writer = imageio.get_writer(
    OUT_MP4,
    fps=OUT_FPS,
    codec="libx264",
    quality=8,
    macro_block_size=None
)

# ---------------- Main loop ----------------
try:
    for i in range(TEST_FRAMES):
        fid = TEST_START_IDX + i

        gt, optical = load_gt_points_and_optical(fid)
        optical_t = torch.tensor(optical, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,30)

        with torch.no_grad():
            pred = model(optical_t).squeeze(0).detach().cpu().numpy().astype(np.float32)

        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

        ax1 = fig.add_subplot(121, projection="3d")
        ax1.scatter(
            gt[:, 0], gt[:, 1], -gt[:, 2],
            s=POINT_SIZE, c=gt[:, 2], cmap="viridis",
            vmin=vmin, vmax=vmax
        )
        ax1.set_title(f"GT  (frame {fid})", fontsize=12)
        set_axes_common(ax1, xlim, ylim, zlim)

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(
            pred[:, 0], pred[:, 1], -pred[:, 2],
            s=POINT_SIZE, c=pred[:, 2], cmap="viridis",
            vmin=vmin, vmax=vmax
        )
        ax2.set_title(f"PR  (frame {fid})", fontsize=12)
        set_axes_common(ax2, xlim, ylim, zlim)

        fig.tight_layout()
        writer.append_data(render_figure_to_rgb(fig))
        plt.close(fig)

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{TEST_FRAMES}] appended frames...")

finally:
    writer.close()

print(f"Saved video to: {OUT_MP4}")
