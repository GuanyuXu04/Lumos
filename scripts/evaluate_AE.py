import os
import argparse
import numpy as np
import yaml
import torch
from lumos.model import PCAutoencoder
from lumos.data import WaveguideDataset
from lumos.metrics import F_score
from lumos.viz import plot_fscore_vs_tau, plot_pc_pair


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run-name", default=None, help="Override output.run_name")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name

    data_cfg   = cfg["data"]
    model_cfg  = cfg["model"]
    out_cfg    = cfg["output"]

    run_dir    = os.path.join(out_cfg["base_dir"], out_cfg["run_name"])
    model_path = os.path.join(run_dir, "checkpoints", "best_model.pth")
    out_dir    = os.path.join(run_dir, "eval_ae")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PCAutoencoder(
        latent_dim=model_cfg["latent_dim"],
        num_points=model_cfg["num_points"],
        tnet1=model_cfg["tnet1"],
        tnet2=model_cfg["tnet2"],
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from {model_path}")

    dataset = WaveguideDataset(data_cfg["path"], stride=data_cfg["stride"])
    print(f"Dataset size: {len(dataset)}")

    # F-score vs tau
    n_samples = min(1000, len(dataset))
    sample_indices = np.random.choice(len(dataset), n_samples, replace=False)
    tau_range = np.arange(0, 3.1, 0.1)

    f_scores = []
    for tau in tau_range:
        f_s = []
        for i in sample_indices:
            origin, _, _ = dataset[int(i)]
            origin = np.asarray(origin, dtype=np.float32)
            origin_tensor = torch.tensor(origin).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(origin_tensor).squeeze(0).cpu().numpy()
            f_s.append(F_score(origin, pred, tau=tau))
        f_scores.append(np.mean(f_s))

    plot_fscore_vs_tau(
        tau_range, f_scores,
        save_path=os.path.join(out_dir, "average_fscore_vs_tau.png"),
    )

    # Visualize 10 evenly-spaced samples
    TAU_VIZ = 2.0
    n_frames = 10
    step = max(1, len(dataset) // n_frames)
    for i in range(n_frames):
        frame = i * step
        pc, _, _ = dataset[frame]
        pc = np.asarray(pc, dtype=np.float32)
        pc_tensor = torch.tensor(pc).unsqueeze(0).to(device)
        with torch.no_grad():
            reconstructed_pc = model(pc_tensor).squeeze(0).cpu().numpy()
        f_score_val = F_score(pc, reconstructed_pc, tau=TAU_VIZ)
        print(f"Frame {frame}: F-score = {f_score_val:.4f}")
        plot_pc_pair(
            pc, reconstructed_pc,
            save_path=os.path.join(out_dir, f"reconstruction_vs_original_{frame}.png"),
            left_title=f"Original (Frame {frame})",
            right_title=f"Reconstructed (Frame {frame})\nF-score @ {TAU_VIZ:.1f}mm: {f_score_val:.4f}",
        )
