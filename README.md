# Lumos

Lumos is a PyTorch reconstruction pipeline for a highly deformable optical-waveguide membrane sensor. The sensor embeds LEDs and photodiodes (PDs) in a soft transparent membrane. LEDs are activated sequentially, PD readings are collected as optical features, and the learned model reconstructs the membrane's 3D geometry in real time.

The current pipeline follows a two-stage design:

1. **Point-cloud autoencoder**: learns a compact latent representation of membrane geometry from RealSense depth-derived point clouds.
2. **PD-to-latent regression**: learns a mapping from optical readings to that latent representation. At inference time, the trained decoder converts the predicted latent vector into a reconstructed 3D point cloud.

---

## Repository layout

```text
Lumos/
├── config/
│   └── config.yaml          # Training, model, data, and output configuration
├── lumos/
│   ├── config.py            # Hardware and feature-selection configuration
│   ├── data.py              # Dataset loading, depth-to-point-cloud conversion, feature selection
│   ├── ddp.py               # Distributed-training helpers
│   ├── losses.py            # Chamfer distance and repulsion loss
│   ├── metrics.py           # Evaluation metrics
│   ├── model.py             # Autoencoder, PD-to-latent network, combined ShapeNet model
│   └── viz.py               # Visualization helpers
├── scripts/
│   ├── train_AE.py          # Train point-cloud autoencoder
│   ├── evaluate_AE.py       # Evaluate point-cloud autoencoder
│   ├── train_optical.py     # Train optical PD-to-latent model
│   └── evaluate.py          # Evaluate final optical-to-shape model
└── tools/
    ├── collect_data.py      # Collect paired RealSense depth + optical readings
    └── interface.py         # Real-time serial inference and visualization
```

---

## Installation

Create and activate a Python environment first:

```bash
conda create -n lumos
conda activate lumos
```

Install PyTorch following the command for your CUDA version from the official PyTorch installation page. Then install this package in editable mode:

```bash
cd /path/to/Lumos
pip install -e .
```

The current `pyproject.toml` only declares the package and `pyyaml`, so install the runtime packages used by the training and tools manually:

```bash
pip install numpy scipy matplotlib opencv-python pyserial open3d
```

For data collection with an Intel RealSense camera, also install:

```bash
pip install pyrealsense2
```

If `open3d` fails to install, check that your Python version is supported by the Open3D wheel available for your OS.

---

## Dataset format

Training expects a folder like `Data/Combined_Data` containing paired optical and depth files:

```text
Data/Combined_Data/
├── meta.npz
├── frame_000000_depth.npy
├── frame_000000_optical.npy
├── frame_000001_depth.npy
├── frame_000001_optical.npy
└── ...
```

Each depth file is a cropped RealSense depth frame. `meta.npz` stores the ROI mask, camera intrinsics, depth scale, labels, and data-collection settings. `lumos.data.WaveguideDataset` converts each depth frame into a point cloud in millimeters using the ROI mask, depth scale, camera intrinsics, and the configured spatial stride.

Each optical row is expected to contain:

```text
time_ms, L0P0, L0P1, ..., L0P6, L1P0, ..., L30P6
```

For the default hardware, this is:

```text
1 + (30 + 1) * (6 + 1) = 218 values
```

Here, index `0` on the LED and PD axes is the off/baseline channel. Active sensing features are normally selected from `L1..L30` and `P1..P6`, giving:

```text
30 LEDs * 6 PDs = 180 optical input features
```

The dataset returns three values:

```python
points, optical, optical_norm
```

The current training scripts use `points` and raw `optical`. The min-max normalized `optical_norm` is computed and returned for future use or experiments.

---

## How the model works

### Stage 1: point-cloud autoencoder

`PCAutoencoder` in `lumos/model.py` consists of:

- `ShapeEncoder`: a PointNet-style encoder that maps an input point cloud `(B, N, 3)` into a latent vector `(B, latent_dim)`.
- `ShapeDecoder`: a decoder that maps the latent vector back into a reconstructed point cloud `(B, num_points, 3)`.

The encoder uses shared point-wise MLP layers implemented as `Conv1d`, optional T-Net alignment modules, global max pooling, and a final latent pooling stage. The decoder uses transposed-convolution heads for common point counts such as 1024, 2048, 4096, and 8192 points. If `num_points` is not exactly one of those supported values, the decoder uses the largest supported transposed-convolution branch below the target and adds the remaining points through an FC branch.

The autoencoder is trained with:

```text
loss = Chamfer distance + repulsion loss
```

The Chamfer term encourages the reconstructed point cloud to match the target point cloud. The repulsion term discourages degenerate point collapse and improves point distribution.

### Stage 2: PD-to-latent regression

After the autoencoder is trained, `train_optical.py` freezes the autoencoder encoder and precomputes a latent vector for each training point cloud:

```text
point cloud -> trained encoder -> latent vector
```

Then `PD2Latent` learns:

```text
optical PD vector -> latent vector
```

The default optical input has 180 features. The regression target is the latent vector produced by the trained point-cloud encoder. The model is trained with MSE loss in latent space.

### Final real-time model

After `PD2Latent` is trained, `train_optical.py` combines:

```text
PD2Latent + trained ShapeDecoder -> ShapeNet
```

The saved combined checkpoint is:

```text
checkpoints/<run_name>/best_combined.pth
```

At inference time:

```text
serial optical readings -> selected optical columns -> PD2Latent -> decoder -> reconstructed 3D point cloud
```

---

## Hardware Configuration: `lumos/config.py`

`lumos/config.py` describes the physical hardware and feature selection. Edit this file when the sensor layout or selected LED/PD features change.

Important fields:

| Field | Meaning |
|---|---|
| `num_leds` | Number of active LEDs in the hardware. Default: `30`. |
| `num_pds` | Number of active photodiodes. Default: `6`. |
| `x_range`, `y_range`, `z_range` | Valid point-cloud bounding box in millimeters. Samples outside this box are filtered out. |
| `FeatureSelection.leds` | LED indices used as model input. Use `1..num_leds` for active LEDs. |
| `FeatureSelection.pds` | PD indices used as model input. Use `1..num_pds` for active PDs. |

Rules to keep configurations consistent:

- The optical row length should be `1 + (num_leds + 1) * (num_pds + 1)`.
- Index `0` is reserved for the off/baseline channel.
- `model.optical_dim` in `config/config.yaml` must equal `len(FeatureSelection.leds) * len(FeatureSelection.pds)`.
- If you change LED/PD selection, also update `COLUMNS` in `tools/interface.py` for real-time inference.

Example: using 15 LEDs and 6 PDs:

```python
@dataclass(frozen=True)
class FeatureSelection:
    leds: Tuple[int, ...] = tuple(range(1, 16))
    pds: Tuple[int, ...] = tuple(range(1, 7))
```

Then set this in `config/config.yaml`:

```yaml
model:
  optical_dim: 90
```

---

## Training Configuration: `config/config.yaml`

`config/config.yaml` mainly controls dataset paths, training hyperparameters, model size, and output locations.

Key fields:

| Section | Field | Meaning |
|---|---|---|
| `data` | `path` | Dataset folder containing `meta.npz` and paired `frame_*_depth.npy` / `frame_*_optical.npy` files. |
| `data` | `stride` | Spatial downsampling stride when converting depth images to point clouds. Larger stride gives fewer points and faster loading. |
| `data` | `train_split`, `val_split` | Dataset split ratios. The remaining samples are used as test data. |
| `model` | `optical_dim` | Number of optical input features. Must match selected LED/PD features. |
| `model` | `latent_dim` | Autoencoder latent dimension and PD-to-latent output dimension. |
| `model` | `num_points` | Number of reconstructed 3D points. Common values: 1024, 2048, 4096, 8192. |
| `model` | `tnet1`, `tnet2` | Enable or disable PointNet-style input/feature transform nets. |
| `output` | `base_dir` | Root directory for checkpoints and evaluation outputs. |
| `output` | `run_name` | Name of the experiment subfolder. |
| `train_ae` | `gpus` | `auto` uses all available GPUs; set to `1` for single GPU. |
| `train_ae` | `loss.repulsion_k`, `loss.repulsion_h` | Repulsion-loss neighborhood size and kernel width. |
| `train_optical` | `ae_run_name` | Optional run name from which to load the trained autoencoder. If `null`, uses `output.run_name`. |

Run outputs are written to:

```text
checkpoints/<run_name>/
```

---

## Running the training and evaluation scripts

All commands below assume you are in the repository root.

### 1. Train the point-cloud autoencoder

```bash
python scripts/train_AE.py --run-name exp1
```

This trains `PCAutoencoder` on depth-derived point clouds. It saves:

```text
checkpoints/membrane_4096/checkpoints/best_model.pth
checkpoints/membrane_4096/record_log.txt
checkpoints/membrane_4096/figs/
```

For multi-GPU training, leave `train_ae.gpus: auto` in `config/config.yaml`, or launch explicitly with `torchrun`:

```bash
torchrun --nproc_per_node=2 scripts/train_AE.py --config config/config.yaml --run-name membrane_4096
```
Note: 
- This process may consume lots of GPU memory, reduce batch size if OOM occur. 
- Training the `PCAutoencoder` is time-consuming. Don't train it repeatedly!


### 2. Train the optical PD-to-latent model

```bash
python scripts/train_optical.py --run-name exp1
```

This loads the trained autoencoder from:

```text
checkpoints/membrane_4096/checkpoints/best_model.pth
```

Then it precomputes latent vectors, trains `PD2Latent`, and saves:

```text
checkpoints/membrane_4096/optical_checkpoints/model_ep*.pth
checkpoints/membrane_4096/best_combined.pth
```

If the autoencoder was trained under a different run name, use:

```bash
python scripts/train_optical.py \
  --run-name optical_run \
  --ae-run-name membrane_4096
```

This writes optical-model outputs to `checkpoints/optical_run/` but loads the autoencoder from `checkpoints/membrane_4096/`.

---

## Collecting paired depth + optical data

`tools/collect_data.py` records paired RealSense depth frames and optical sensor readings. It is currently configured through constants at the top of the file rather than command-line arguments.

### Hardware assumptions

The default script assumes:

- Intel RealSense D435 depth stream.
- Serial optical sensor stream over a COM port.
- One newline-terminated CSV row per optical scan.
- Default optical row length: `218` values.
- Data saved into `Combined_Data/`.

The membrane sensor controller should output rows in this order (see this repo: [https://github.com/XuGuaaaanyu/Optical_Tomography](https://github.com/XuGuaaaanyu/Optical_Tomography) ):

```text
time_ms, L0P0, L0P1, ..., L0P6, L1P0, ..., L30P6
```

### Configure the collection script

Open `tools/collect_data.py` and edit the user settings:

```python
SERIAL_PORT = "COM6"
BAUD_RATE = 2_000_000
SER_TIMEOUT = 0.0
EXPECTED_VALS = 218
SESSION_DIR = Path("Combined_Data")
COMBINED_DIR = SESSION_DIR
```

Also check the RealSense settings:

```python
DEPTH_W, DEPTH_H, DEPTH_FPS = 848, 480, 30
ENABLE_EMITTER = True
LASER_POWER_ABS = 360.0
USE_MANUAL_EXPOSURE = True
MANUAL_EXPOSURE_US = 1000.0
MANUAL_GAIN = 16.0
TRY_SET_DEPTH_UNITS = True
DEPTH_UNITS_VALUE = 0.0005
```

On Linux, the serial port will usually look like `/dev/ttyACM0` or `/dev/ttyUSB0` instead of `COM6`.

### Run data collection

```bash
python tools/collect_data.py
```

On the first run, the RealSense preview window shows a fixed central square ROI. Align the membrane inside the square and press Enter to start logging. Press `q`, `Esc`, or use `Ctrl+C` to stop.

The script saves:

```text
Combined_Data/meta.npz
Combined_Data/frame_000000_depth.npy
Combined_Data/frame_000000_optical.npy
Combined_Data/frame_000001_depth.npy
Combined_Data/frame_000001_optical.npy
...
```

Important behavior:

- The script scans existing files and continues from the next available global frame ID.
- If `meta.npz` already exists, it reuses the saved ROI and camera settings to keep the session consistent.
- If you change camera settings, ROI, or sensor hardware, start a new data folder or remove the old session data.


---

## Real-time reconstruction interface

Coming soon ...

---

## Citation

For details of the sensor system and reconstruction pipeline, see the project paper:

```text
@article{xu2026highly,
  title={Highly Deformable Proprioceptive Membrane for Real-Time 3D Shape Reconstruction},
  author={Xu, Guanyu and Wang, Jiaqi and Tong, Dezhong and Huang, Xiaonan},
  journal={arXiv preprint arXiv:2601.13574},
  year={2026}
}
```
