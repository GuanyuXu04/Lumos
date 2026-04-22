import os
import numpy as np
import matplotlib.pyplot as plt


def set_axes_equal(ax1, ax2):
    """Rescale two 3D axes to a shared cubic bounding box sized by ax1's current limits."""
    x_limits = ax1.get_xlim3d()
    y_limits = ax1.get_ylim3d()
    z_limits = ax1.get_zlim3d()
    max_range = max(
        abs(x_limits[1] - x_limits[0]),
        abs(y_limits[1] - y_limits[0]),
        abs(z_limits[1] - z_limits[0]),
    )
    xm = sum(x_limits) / 2
    ym = sum(y_limits) / 2
    zm = sum(z_limits) / 2
    for ax in (ax1, ax2):
        ax.set_xlim3d([xm - max_range / 2, xm + max_range / 2])
        ax.set_ylim3d([ym - max_range / 2, ym + max_range / 2])
        ax.set_zlim3d([zm - max_range / 2, zm + max_range / 2])


def plot_fscore_vs_tau(tau_range, f_scores, save_path, title="Average F-score vs Tau"):
    """Plot F-score vs tau curve and save to save_path."""
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(tau_range, f_scores, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Tau (mm)")
    ax.set_ylabel("Average F-score")
    ax.grid()
    fig.savefig(save_path)
    plt.close(fig)


def plot_pc_pair(gt_pc, pred_pc, save_path, left_title="Ground Truth", right_title="Reconstructed"):
    """Side-by-side 3D scatter plots of two point clouds, coloured by Z depth.

    Args:
        gt_pc:        (N, 3) array-like, ground truth points (mm)
        pred_pc:      (M, 3) array-like, predicted points (mm)
        save_path:    output file path
        left_title:   title for the ground truth subplot
        right_title:  title for the reconstructed subplot
    """
    gt_pc   = np.asarray(gt_pc,   dtype=np.float32)
    pred_pc = np.asarray(pred_pc, dtype=np.float32)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(gt_pc[:, 0], gt_pc[:, 1], -gt_pc[:, 2], s=1, c=gt_pc[:, 2], cmap="viridis")
    ax1.set_title(left_title, fontsize=12)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(pred_pc[:, 0], pred_pc[:, 1], -pred_pc[:, 2], s=1, c=pred_pc[:, 2], cmap="viridis")
    ax2.set_title(right_title, fontsize=12)
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_zlabel("Z (mm)")

    set_axes_equal(ax1, ax2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
