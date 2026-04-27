import os
import random
import argparse
import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
from lumos.model import ShapeNet
from lumos.data import WaveguideDataset
from lumos.metrics import F_score_batched, chamfer_distance_batched, F_score
from lumos.viz import plot_fscore_vs_tau, plot_pc_pair
from torch.utils.data import DataLoader, Subset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run-name", default=None, help="Override output.run_name")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    out_cfg   = cfg["output"]

    run_dir    = os.path.join(out_cfg["base_dir"], out_cfg["run_name"])
    model_path = os.path.join(run_dir, "best_combined.pth")
    out_dir    = os.path.join(run_dir, "eval")
    fig_dir    = os.path.join(out_dir, "figures")
    fscore_dir = os.path.join(out_dir, "fscore")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(fscore_dir, exist_ok=True)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Using device {device}")
    model = ShapeNet(
        pd_in_features=model_cfg["optical_dim"],
        latent_dim=model_cfg["latent_dim"],
        num_points=model_cfg["num_points"],
        tnet1=model_cfg["tnet1"],
        tnet2=model_cfg["tnet2"],
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.pd2latent.load_state_dict(checkpoint["pd2latent_state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    model.eval()
    print(f"Model loaded from {model_path}")

    dataset = WaveguideDataset(data_cfg["path"], stride=data_cfg["stride"])
    n_total = len(dataset)
    n_train = int(data_cfg["train_split"] * n_total)
    n_val   = int(data_cfg["val_split"] * n_total)
    n_test  = n_total - n_train - n_val

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    test_subset = Subset(test_set, list(range(min(500, len(test_set)))))
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    print(f"Test set size: {len(test_subset)} samples")

    # ------------------------------
    # Inference on test set
    # ------------------------------
    all_gt, all_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            points_gt, batch_optical, _ = batch
            points_pred = model(batch_optical.to(device).float()).cpu().numpy()
            all_gt.append(points_gt.numpy())
            all_pred.append(points_pred)

    all_gt   = np.concatenate(all_gt, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    # ------------------------------
    # Visualize ground truth vs reconstructed
    # ------------------------------
    TAU_VIZ = 1.5
    sample_indices = random.sample(range(len(test_set)), min(100, len(test_set)))
    for idx in sample_indices:
        origin_pc, optical, _ = test_set[idx]
        optical_tensor = torch.tensor(optical, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            pred_pc = model(optical_tensor).squeeze(0).cpu().numpy()
        f_score = F_score(origin_pc, pred_pc, tau=TAU_VIZ)
        print(f"Sample {idx}: F-score @ {TAU_VIZ} mm = {f_score:.4f}")
        plot_pc_pair(
            origin_pc, pred_pc,
            save_path=os.path.join(fig_dir, f"PR_vs_GT_{idx}.svg"),
            left_title=f"Ground Truth Point Cloud (Sample {idx})",
            right_title=f"Reconstructed Point Cloud (Sample {idx})\nF-score @ {TAU_VIZ:.1f}mm: {f_score:.4f}",
        )

    # ------------------------------
    # F-score vs tau
    # ------------------------------
    tau_range = np.arange(0, 3.1, 0.1)
    f_scores, results_lines = [], []
    with torch.no_grad():
        for tau in tau_range:
            per_item = []
            for points_gt, batch_optical, _ in test_loader:
                points_pred_t = model(batch_optical.to(device).float())
                fs_b = F_score_batched(points_pred_t, points_gt.to(device).float(), tau)
                per_item.append(fs_b.detach().cpu())
            avg_f = float(torch.cat(per_item).numpy().mean())
            f_scores.append(avg_f)
            print(f"Tau: {tau:.2f} mm, Average F-score: {avg_f:.4f}")
            results_lines.append(f"{tau:.2f}, {avg_f:.4f}")

    with open(os.path.join(fscore_dir, "F_score_data_record.txt"), "w") as f:
        f.write("tau, average_fscore\n")
        f.writelines(line + "\n" for line in results_lines)

    plot_fscore_vs_tau(
        tau_range, f_scores,
        save_path=os.path.join(fscore_dir, "average_fscore_vs_tau.png"),
    )

    # ------------------------------
    # Chamfer distance
    # ------------------------------
    cd_list = []
    with torch.no_grad():
        for points_gt, batch_optical, _ in test_loader:
            points_pred = model(batch_optical.to(device).float())
            cd_list.append(chamfer_distance_batched(points_pred, points_gt.to(device).float()).detach().cpu())

    cd_all = torch.cat(cd_list).numpy()
    print(f"[Chamfer] mean = {cd_all.mean():.6f}, std = {cd_all.std(ddof=0):.6f}")

    with open(os.path.join(out_dir, "hist_data_chamfer.txt"), "w") as f:
        f.writelines(f"{v:.6f}\n" for v in cd_all)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(cd_all, bins=40)
    ax.set_title("Chamfer Distance Histogram (Euclidean)")
    ax.set_xlabel("Chamfer distance")
    ax.set_ylabel("Count")
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_chamfer.png"), dpi=200)
    plt.close(fig)
