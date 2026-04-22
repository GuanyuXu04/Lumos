from pathlib import Path
from typing import Tuple, List
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

# ---------- helpers ----------
def _load_meta(meta_path: str):
    Z = np.load(meta_path, allow_pickle=True)
    roi_mask = Z["roi_mask"].astype(bool)
    scale_m  = float(Z["depth_scale_m_per_unit"])

    if "depth_K_roi" in Z:
        K = Z["depth_K_roi"].astype(float)
    else:
        if "depth_K_full" not in Z or "roi_origin_xy" not in Z:
            raise KeyError("Missing intrinsics in meta.npz")
        K_full = Z["depth_K_full"].astype(float)
        x0, y0 = Z["roi_origin_xy"]
        K = K_full.copy()
        K[0, 2] -= float(x0)
        K[1, 2] -= float(y0)
    return roi_mask, scale_m, K


def _depth_to_points_mm_fixedgrid(depth_u16, mask, K, scale_m, stride=2):
    h, w = depth_u16.shape
    # --- build index grid (supports integer or float stride) ---
    if isinstance(stride, (float, np.floating)) and not float(stride).is_integer():
        # target size = round(original / stride)
        tw = int(round(w / float(stride)))
        th = int(round(h / float(stride)))
        # evenly spaced integer indices via linspace -> round (endpoint=False avoids duplication)
        u_idx = np.round(np.linspace(0, w - 1, num=tw, endpoint=False)).astype(np.int32)
        v_idx = np.round(np.linspace(0, h - 1, num=th, endpoint=False)).astype(np.int32)
    else:
        s = int(stride)
        u_idx = np.arange(0, w, s, dtype=np.int32)
        v_idx = np.arange(0, h, s, dtype=np.int32)

    uu, vv = np.meshgrid(u_idx.astype(np.float32), v_idx.astype(np.float32), indexing='xy')

    depth = depth_u16[np.ix_(v_idx, u_idx)].astype(np.float32)
    m     = mask[np.ix_(v_idx, u_idx)]

    # Gaussian smoothing
    ksize, sigma = 11, 1.2
    truncate = ((ksize - 1) / 2) / sigma  # solve for truncate so kernel matches size
    depth = gaussian_filter(depth, sigma=sigma, truncate=truncate)
    uu = uu[m]; vv = vv[m]; z_m = depth[m] * scale_m

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x_m = (uu - cx) * z_m / fx
    y_m = (vv - cy) * z_m / fy
    pts_mm = np.column_stack([x_m, y_m, z_m]).astype(np.float32) * 1000.0

    if pts_mm.shape[0] > 0:
        pts_mm[:, 0] -= pts_mm[0, 0]
        pts_mm[:, 1] -= pts_mm[0, 1]
    
    #print(pts_mm)
    return pts_mm


# ---------- Dataset ----------
class WaveguideDataset(Dataset):
    def __init__(self, folder: str, stride: int = 1, x_range=(-10,200), y_range=(-10,200), z_range=(180,250), verbose: bool = True):
        self.folder = Path(folder)
        self.roi_mask, self.scale_m, self.K = _load_meta(self.folder / "meta.npz")

        self.stride = stride
        self.x_range, self.y_range, self.z_range = x_range, y_range, z_range
        self.verbose = verbose

        # Pair depth + optical
        self.depth_files = sorted(self.folder.glob("frame_*_depth.npy"))
        self.optical_files = [self.folder / f.stem.replace("_depth","_optical.npy") for f in self.depth_files]

        self._cols = np.array([13 + 7 * j for j in range(30)], dtype=np.int64)

        # Cache paths
        self.valid_idx_path = self.folder / "valid_idx.txt"
        self.optical_minmax_path = self.folder / "optical_min_max.npy"

        self.valid_indices = self._load_or_build_valid_indices()
        self.col_min, self.col_max = self._load_or_build_minmax()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _load_or_build_valid_indices(self):
        valid_indices = []
        if self.valid_idx_path.exists():
            # Use cached indices (one integer per line)
            try:
                cached = [int(line.strip()) for line in self.valid_idx_path.read_text().splitlines() if line.strip()]
            except Exception:
                cached = []

            self._log(f"Loaded {len(cached)} cached valid indices")

            # Keep only indices that still make sense (file exists, in-range)
            n = len(self.depth_files)
            for i in cached:
                if 0 <= i < n and self.optical_files[i].exists():
                    valid_indices.append(i)
            # If cache gave us something usable, stop here
            if len(valid_indices) > 0:
                return valid_indices

        # Otherwise, compute valid indices from scratch
        self._log("No cached valid indices found. Building valid indices from scratch...")
        for i, ofile in enumerate(self.optical_files):
            if not ofile.exists():
                continue
            depth = np.load(self.depth_files[i], mmap_mode="r")
            pts = _depth_to_points_mm_fixedgrid(depth, self.roi_mask, self.K, self.scale_m, stride=self.stride)
            ########################################################
            if i < 161489:
                pts *= 2.0
            ########################################################
            
            if pts.shape[0] == 0:
                continue
            x_valid = np.all((self.x_range[0] <= pts[:, 0]) & (pts[:, 0] <= self.x_range[1]))
            y_valid = np.all((self.y_range[0] <= pts[:, 1]) & (pts[:, 1] <= self.y_range[1]))
            z_valid = np.all((self.z_range[0] <= pts[:, 2]) & (pts[:, 2] <= self.z_range[1]))
            if x_valid and y_valid and z_valid:
                valid_indices.append(i)
        
        # Cache valid indices
        try:
            self.valid_idx_path.write_text("\n".join(str(i) for i in valid_indices))
        except Exception:
            pass
        return valid_indices
    
    def _load_or_build_minmax(self):
        if self.optical_minmax_path.exists():
            try:
                arr = np.load(self.optical_minmax_path)
                if isinstance(arr, np.ndarray) and arr.shape == (2, len(self._cols)):
                    self._log("Loaded cached optical min/max")
                    col_min, col_max = arr[0].astype(np.float32), arr[1].astype(np.float32)
                    return col_min, col_max
            except Exception:
                pass
        
        self._log("No cached optical min/max found. Building from scratch...")
        if len(self.valid_indices) == 0:
            col_min = np.zeros((180,), dtype=np.float32)
            col_max = np.ones((180,), dtype=np.float32)
            return col_min, col_max

        col_min = np.full((180,), np.inf, dtype=np.float32)
        col_max = np.full((180,), -np.inf, dtype=np.float32)
        for i in self.valid_indices:
            x = np.load(self.optical_files[i]).astype(np.float32)
            vec = np.asarray(x, dtype=np.float32)[self._cols]

            m = np.isfinite(vec)
            if not np.all(m):
                vmin = np.where(m, vec, np.inf)
                vmax = np.where(m, vec, -np.inf)
            else:
                vmin, vmax = vec, vec
            
            col_min = np.minimum(col_min, vmin)
            col_max = np.maximum(col_max, vmax)
        
        col_min[~np.isfinite(col_min)] = 0.0
        col_max[~np.isfinite(col_max)] = 1.0

        col_min = col_min.astype(np.float32)
        col_max = col_max.astype(np.float32)

        # Cache min/max
        try:
            np.save(self.optical_minmax_path, np.stack([col_min, col_max], axis=0))
        except Exception:
            pass

        return col_min, col_max
    

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        i = self.valid_indices[idx]
        depth = np.load(self.depth_files[i], mmap_mode="r")
        optical_full = np.load(self.optical_files[i])
        optical = optical_full[self._cols].astype(np.float32)

        pts = _depth_to_points_mm_fixedgrid(depth, self.roi_mask, self.K, self.scale_m,stride=self.stride)
        ########################################################
        if i < 161489:
            pts *= 2.0
        ########################################################
        denom = (self.col_max - self.col_min).astype(np.float32)
        safe_denom = np.where(denom > 0.0, denom, 1.0).astype(np.float32)
        optical_norm = (optical - self.col_min.astype(np.float32)) / safe_denom
        optical_norm = np.clip(optical_norm, 0.0, 1.0)

        return pts.astype(np.float32), optical, optical_norm
