import os
import torch
import random
import numpy as np
from lumos.model import ShapeNet
import matplotlib.pyplot as plt
from lumos.data import WaveguideDataset
from lumos.metrics import F_score_batched, chamfer_distance_batched, F_score
from torch.utils.data import DataLoader
from torch.utils.data import Subset

NUM_POINTS = 4096
LATENT_DIM = 512
STRIDE = 3
DATA_PATH = "Data/Combined_Data"
MODEL_PATH = "checkpoints/best_combined_500.pth"
OUT_DIR = "evaluation_results"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ShapeNet(pd_in_features=180, latent_dim=LATENT_DIM, num_points=NUM_POINTS, tnet1=False, tnet2=False).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.pd2latent.load_state_dict(checkpoint['pd2latent_state_dict'])
model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
model.eval()

print(f"Model loaded from {MODEL_PATH}")

dataset = WaveguideDataset(DATA_PATH, stride=STRIDE)
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val   = int(0.1 * n_total)
n_test  = n_total - n_train - n_val

train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(56)
)

test_subset = Subset(test_set, list(range(500)))
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
print(f"Test set size: {len(test_subset)} samples")


# ------------------------------
# Inference on test set
# ------------------------------
all_gt = []
all_pred = []
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        points_gt, batch_optical, _ = batch
        optical_tensor = batch_optical.to(device).float()  # (B, 150)
        points_pred = model(optical_tensor).cpu().numpy()  # (B, N, 3)

        all_gt.append(points_gt.numpy())
        all_pred.append(points_pred)

all_gt = np.concatenate(all_gt, axis=0)
all_pred = np.concatenate(all_pred, axis=0)


# ------------------------------
# Visualize original vs reconstructed point clouds
# ------------------------------
all_indices = list(range(len(test_set)))
sample_indices = random.sample(all_indices, 100)  # pick 100 random samples
for idx in sample_indices:
    origin_pc, optical,_ = test_set[idx]
    optical_tensor = torch.tensor(optical, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        pred_pc = model(optical_tensor).squeeze(0).cpu().numpy()
        print(pred_pc.shape)  # (N, 3)

    f_score = F_score(origin_pc, pred_pc, tau=1.5)
    print(f"Sample {idx}: F-score @ 1.5 mm = {f_score:.4f}")

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')

    ax.scatter(origin_pc[:,0], origin_pc[:,1], -origin_pc[:,2], cmap='viridis', s=1, c=origin_pc[:,2])
    #ax.plot_trisurf(pred_pc[:,0], pred_pc[:,1], pred_pc[:,2],
    #           color='red', alpha=0.5, label='Reconstructed')

    ax.set_title(f'Ground Truth Point Cloud (Sample {idx})', fontsize=12)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.axis('equal')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pred_pc[:,0], pred_pc[:,1], -pred_pc[:,2], cmap='viridis', s=1, c=pred_pc[:,2])
    ax2.set_title(f'Reconstructed Point Cloud (Sample {idx})\nF-score @ 1.5mm: {f_score:.4f}',fontsize=12)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    plt.axis('equal')
    plt.savefig(f'evaluation_results/figures/PR_vs_GT_{idx}.svg', dpi=200)
    plt.close()

# ------------------------------
# Evaluate F-score over range of tau
# ------------------------------
tau_range = np.arange(0, 3.1, 0.1)
f_scores = []
results_lines = []
with torch.no_grad():
    for tau in tau_range:
        per_item = []
        for points_gt, batch_optical, _ in test_loader:
            points_gt_t = points_gt.to(device).float()  # (B, N, 3)
            optical_tensor = batch_optical.to(device).float()  # (B, 180)
            points_pred_t = model(optical_tensor)  # (B, N, 3)
            fs_b = F_score_batched(points_pred_t, points_gt_t, tau)  # (B,)
            per_item.append(fs_b.detach().cpu())
        
        all_fs = torch.cat(per_item, dim=0).numpy()  # (total_samples,)
        avg_f = all_fs.mean()
        f_scores.append(avg_f)
        print(f"Tau: {tau:.2f} mm, Average F-score: {avg_f:.4f}")
        line = f"{tau:.2f}, {avg_f:.4f}"
        results_lines.append(line)

os.makedirs("results/1031_results", exist_ok=True)
with open("results/1031_results/F_score_data_record.txt", "w") as f:
    f.write("tau, average_fscore\n")
    for line in results_lines:
        f.write(line + "\n")

plt.figure(figsize=(8,6))
plt.plot(tau_range, f_scores, marker='o')
plt.title('Average F-score vs Tau')
plt.xlabel('Tau (mm)')
plt.ylabel('Average F-score')
plt.grid()
plt.savefig('results/1031_results/average_fscore_vs_tau_combined.png')
#plt.show()

# ------------------------------
# Evaluate Chamfer
# ------------------------------
cd_list = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        points_gt, batch_optical, _ = batch                               # points_gt: (B, N, 3)
        points_gt_t = points_gt.to(device).float()                      # (B, N, 3)
        optical_t = batch_optical.to(device).float()               # (B, 150)
        points_pred = model(optical_t)

        cd_b = chamfer_distance_batched(points_pred, points_gt_t)  # (B,)
        cd_list.append(cd_b.detach().cpu())

cd_all = torch.cat(cd_list, dim=0).numpy() 
cd_mean = float(cd_all.mean())
cd_std  = float(cd_all.std(ddof=0))
print(f"[Chamfer] mean = {cd_mean:.6f}, std = {cd_std:.6f}")

with open(os.path.join(OUT_DIR, "hist_data_chamfer.txt"), "w") as f:
    for value in cd_all:
        f.write(f"{value:.6f}\n")

# Histograms
plt.figure(figsize=(8,6))
plt.hist(cd_all, bins=40)
plt.title("Chamfer Distance Histogram (Euclidean)")
plt.xlabel("Chamfer distance")
plt.ylabel("Count")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hist_chamfer.png"), dpi=200)