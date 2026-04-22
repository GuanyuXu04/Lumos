import numpy as np
import pandas as pd
import os
import torch
import random
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from lumos.model import PD2Latent, PCAutoencoder, ShapeNet
from lumos.data import WaveguideDataset
import sage
from sage.grouped_imputers import GroupedMarginalImputer
from sage.permutation_estimator import PermutationEstimator
from lumos.metrics import F_score_batched, chamfer_distance_batched, F_score

torch.backends.cudnn.benchmark = True
random.seed(56)
np.random.seed(56)
torch.manual_seed(56)

DATA_PATH = "Data/Combined_Data/Combined_Data"
BEST_AE_EP = 9
NUM_LEDS = 30
NUM_PDS = 5
OPTICAL_DIM = NUM_LEDS * NUM_PDS
LATENT_DIM = 512
NUM_POINTS = 4096
BEST_AE_PATH = f"experiments/L{LATENT_DIM}/NP{NUM_POINTS}/checkpoints/model_ep{BEST_AE_EP}.pth"
MODEL_PATH = f"experiments/L{LATENT_DIM}/NP{NUM_POINTS}/best_combined_norm.pth"
TNET1 = False  # Whether to use input transform net
TNET2 = False  # Whether to use feature transform net
STRIDE = 3
BATCH_SIZE = 64
OUT_DIR = f"experiments/L{LATENT_DIM}/NP{NUM_POINTS}/sage_results"
dataset = WaveguideDataset(DATA_PATH, stride=STRIDE)
sample_point, sample_optical, _ = dataset[0]
print(f"Sample point cloud shape: {sample_point.shape}")
print(f"Sample optical signal shape: {sample_optical.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae = PCAutoencoder(latent_dim=LATENT_DIM, num_points=NUM_POINTS, tnet1=TNET1, tnet2=TNET2).to(device)

best_ae = torch.load(BEST_AE_PATH, map_location=device)
ae.load_state_dict(best_ae['model_state_dict'])
ae.eval()
print(f"Autoencoder loaded from {BEST_AE_PATH}")

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(56)
)

# Precompute latent vectors for all point clouds
print("Precomputing latent vectors for all point clouds...")
latent_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  ### iterate full dataset
optical_signals = []
latent_vectors = []
with torch.no_grad():
    for batch_idx, batch in enumerate(latent_loader):
        batch_points, _ , batch_optical = batch
        batch_points = batch_points.to(device).float()  # (B, N, 3)
        #print(batch_points.shape)
        z = ae.encoder(batch_points)  # (B, LATENT_DIM)

        latent_vectors.append(z.cpu())
        optical_signals.append(batch_optical)

        if (batch_idx + 1) % 20 == 0:
            print(f"Processed {batch_idx + 1} / {len(latent_loader)} batches")

latent_vectors = torch.cat(latent_vectors, dim=0)   # (N, LATENT_DIM)
optical_filtered = torch.cat(optical_signals, dim=0)  # (N, optical_dim)

print(f"Precomputed latent vectors shape: {latent_vectors.shape}")

# Prepare dataset
X_test = optical_filtered.clone().detach().to(torch.float32)
Z_test = latent_vectors.clone().detach().to(torch.float32)

print(f"Test set shape: X_test {X_test.shape}, Z_test {Z_test.shape}")

model = PD2Latent(in_features=OPTICAL_DIM, out_features=LATENT_DIM).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['pd2latent_state_dict'])
model.eval()

# --------------------------- SAGE ----------------------------
# Step 1: Prepare LED groups, model wrapper
led_groups = [list(range(i * NUM_PDS, (i + 1) * NUM_PDS)) for i in range(NUM_LEDS)]
led_names = [f"LED_{i+1}" for i in range(NUM_LEDS)]

pd_groups = [list(range(i, NUM_LEDS * NUM_PDS, NUM_PDS)) for i in range(NUM_PDS)]
pd_names = [f"PD_{i+1}" for i in range(NUM_PDS)]

def model_predict(x):
    model.eval()
    x_torch = torch.from_numpy(x).to(device).float()
    with torch.no_grad():
        output = model(x_torch)
    return output.cpu().numpy()


# Step 2: Initialize grouped marginal imputer, permutation estimator
rng = np.random.default_rng(seed=56)
idx = rng.choice(len(X_test), size=512, replace=False)
background_data = X_test[idx].numpy()
led_imputer = GroupedMarginalImputer(model_predict, background_data, led_groups)
led_estimator = PermutationEstimator(led_imputer, loss='mse')

pd_imputer = GroupedMarginalImputer(model_predict, background_data, pd_groups)
pd_estimator = PermutationEstimator(pd_imputer, loss='mse', random_state=56)

# Step 3: Compute SAGE values for each LED group
print("Start SAGE calculation for 30 LED groups ...")
led_sage_values = led_estimator(
    X_test.numpy(),
    Z_test.numpy(),
    batch_size=128,
    detect_convergence=True,
    thresh=0.05,
    verbose=True,
    bar=True
)
print("SAGE Values for each LED group:")
led_results = pd.DataFrame({
    'LED_Group': led_names,
    'SAGE_Value': led_sage_values.values,
    'Std_Err': led_sage_values.std
}).sort_values(by='SAGE_Value', ascending=False)
print(led_results)

pd_sage_values = pd_estimator(
    X_test.numpy(),
    Z_test.numpy(),
    batch_size=128,
    detect_convergence=True,
    thresh=0.05,
    verbose=True,
    bar=True
)
pd_results = pd.DataFrame({
    'PD_Group': pd_names,
    'SAGE_Value': pd_sage_values.values,
    'Std_Err': pd_sage_values.std
}).sort_values(by='SAGE_Value', ascending=False)
print(pd_results)

# Step 4: Save the results for plotting

os.makedirs(OUT_DIR, exist_ok=True)
led_csv_path = os.path.join(OUT_DIR, "sage_led_importance.csv")
pd_csv_path = os.path.join(OUT_DIR, "sage_pd_importance.csv")

led_results.to_csv(led_csv_path, index=False)
pd_results.to_csv(pd_csv_path, index=False)

print(f"\nResults successfully saved:")
print(f"1. LED Importance: {led_csv_path}")
print(f"2. PD Importance:  {pd_csv_path}")

