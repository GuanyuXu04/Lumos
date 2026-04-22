import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import random
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from lumos.model import PD2Latent, PCAutoencoder, ShapeNet
from lumos.data import WaveguideDataset

torch.backends.cudnn.benchmark = True
random.seed(56)
np.random.seed(56)
torch.manual_seed(56)

DATA_PATH = "Data/Combined_Data"
BEST_AE_EP = 9
OPTICAL_DIM = 30  # Dimensionality of optical property vector
LATENT_DIM = 512
NUM_POINTS = 4096
BEST_AE_PATH = "checkpoints/model_ep15.pth"
TNET1 = False  # Whether to use input transform net
TNET2 = False  # Whether to use feature transform net
STRIDE = 3
BATCH_SIZE = 64
EPOCHS = 500
LR = 0.001

dataset = WaveguideDataset(DATA_PATH, stride=STRIDE)
sample_point, sample_optical, _ = dataset[0]
print(f"Sample point cloud shape: {sample_point.shape}")
print(f"Sample optical signal shape: {sample_optical.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae = PCAutoencoder(latent_dim=LATENT_DIM, num_points=NUM_POINTS, tnet1=TNET1, tnet2=TNET2).to(device)
model = PD2Latent(in_features=OPTICAL_DIM, out_features=LATENT_DIM).to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_ae = torch.load(BEST_AE_PATH, map_location=device)
ae.load_state_dict(best_ae['model_state_dict'])
ae.eval()
print(f"Autoencoder loaded from {BEST_AE_PATH}")

# Precompute latent vectors for all point clouds
print("Precomputing latent vectors for all point clouds...")
latent_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)  ### iterate full dataset
optical_signals = []
latent_vectors = []
with torch.no_grad():
    for batch_idx, batch in enumerate(latent_loader):
        batch_points, batch_optical, _ = batch
        batch_points = batch_points.to(device).float()  # (B, N, 3)
        #print(batch_points.shape)
        z = ae.encoder(batch_points)  # (B, LATENT_DIM)

        latent_vectors.append(z.cpu())
        optical_signals.append(batch_optical)

        if (batch_idx + 1) % 200 == 0:
            print(f"Processed {batch_idx + 1} / {len(latent_loader)} batches")

latent_vectors = torch.cat(latent_vectors, dim=0)   # (N, LATENT_DIM)
optical_filtered = torch.cat(optical_signals, dim=0)  # (N, optical_dim)

print(f"Precomputed latent vectors shape: {latent_vectors.shape}")

# Prepare dataset and dataloaders
X = torch.tensor(optical_filtered, dtype=torch.float32)  # (N, 150)
Z = torch.tensor(latent_vectors, dtype=torch.float32)  # (N, LATENT_DIM)
dataset = torch.utils.data.TensorDataset(X, Z)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(56)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train_epoch():
    model.train()
    loss_sum = 0.0

    for batch_x, batch_z in train_loader:
        batch_x, batch_z = batch_x.to(device), batch_z.to(device)
        optimizer.zero_grad()
        pred_z = model(batch_x)
        loss = torch.mean((pred_z - batch_z) ** 2)
        #print("Batch Loss:", loss.item())
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * batch_x.size(0)

    return loss_sum / max(1, len(train_loader))

@torch.no_grad()
def evaluate_epoch():
    model.eval()
    loss_sum = 0.0
    for batch_x, batch_z in val_loader:
        batch_x, batch_z = batch_x.to(device), batch_z.to(device)
        pred_z = model(batch_x)
        loss = torch.mean((pred_z - batch_z) ** 2)
        #print("Val Batch Loss:", loss.item())
        loss_sum += loss.item() * batch_x.size(0)
    return loss_sum / max(1, len(val_loader))

def train(patience=20):
    os.makedirs("checkpoints/optical_checkpoints", exist_ok=True)
    best_val = float('inf')
    best_path = None
    epochs_without_improve = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch()
        val_loss = evaluate_epoch()
        sched.step(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        ep_path = os.path.join(f"checkpoints/optical_checkpoints", f"model_ep{epoch+1}.pth")
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improve = 0
            best_path = ep_path
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, best_path)
        else:
            epochs_without_improve += 1
        
        if epochs_without_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    return best_val, best_path

print("Starting training...")
best_val, best_path = train(patience=20)
print(f"Best model saved at {best_path} with Val Loss: {best_val:.6f}")

# Save the combined model
combined_model = ShapeNet(pd_in_features=optical_filtered.shape[1], latent_dim=LATENT_DIM, num_points=NUM_POINTS, tnet1=TNET1, tnet2=TNET2)
pd2latent_checkpoint = torch.load(best_path, map_location=device)
combined_model.pd2latent.load_state_dict(pd2latent_checkpoint['model_state_dict'])
combined_model.decoder.load_state_dict(ae.decoder.state_dict())

os.makedirs(f"checkpoints", exist_ok=True)
combined_path = os.path.join(f"checkpoints", "best_combined_500.pth")

torch.save({
    "epoch": pd2latent_checkpoint.get("epoch", None),
    "val_loss": best_val,
    "pd_dim": optical_filtered.shape[1],
    "latent_dim": LATENT_DIM,
    "num_points": NUM_POINTS,
    "tnet1": TNET1,
    "tnet2": TNET2,
    "pd2latent_state_dict": combined_model.pd2latent.state_dict(),
    "decoder_state_dict": combined_model.decoder.state_dict(),
}, combined_path)

print(f"Combined model saved to {combined_path}")