# spatial_perturbation_maps.py
"""
Complete script to generate and save 2D sensitivity maps for each gauging station,
clipped to the Michigan boundary and overlaid with station locations.
"""
from pathlib import Path
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from rasterio.features import geometry_mask
from affine import Affine
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from typing import Optional, Tuple
# allow project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_loader import HDF5Dataset
from model import CNN_LSTM
from shapely.errors import ShapelyDeprecationWarning
import warnings
from shapely.geometry import Point
from scipy.stats import ks_2samp
from shapely.ops import unary_union
import seaborn as sns
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# -----------------------------------------------------------------------------
# LOAD CONFIG
# -----------------------------------------------------------------------------
CONF_PATH = Path(__file__).resolve().parents[2] / 'config' / 'config.yaml'
with open(CONF_PATH, 'r') as f:
    CONF = yaml.safe_load(f)

# Paths
H5_FILE    = Path(CONF['h5_file'])
LABELS_CSV = Path(CONF['labels_path'])
USGS_METADATA = Path(CONF['usgs_clusters'])
CLUSTERS_CSV   = Path(CONF['usgs_clusters'])  # CSV with site_no, cluster_id
CHECKPOINT = Path(CONF['checkpoint_path'])
SHAPE_PATH = Path(CONF.get('watershed_shp', ''))
BEDROCK_PATH = Path(CONF.get('bedrock_shp', ''))
# after loading CONF:
CACHE_PATH = Path(CONF.get("sensitivity_cache", "sigma_100_sens_maps.npy"))
# Perturbation params
SIGMA_PIX  = tuple(CONF.get('sigma_pix', CONF.get('sigma_pixels', [100, 100])))
AMPLITUDE  = float(CONF.get('amplitude', 1.0))
BATCH_SIZE = int(1)
NUM_WORKERS= int(CONF.get('num_workers', 8))
# Device selection

DEVICE = torch.device(f"cuda:{CONF['gpu'][0]}" if torch.cuda.is_available() else "cpu") 

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def load_gauge_metadata(path: Path) -> pd.DataFrame:
    """Load gauge metadata CSV with 'site_no','lat','lona'."""
    df = pd.read_csv(path, dtype={'station_id': str})
    required = {'station_id', 'lat', 'lon', 'cluster'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Gauge metadata CSV missing columns: {missing}")
    return df.reset_index(drop=True)


def gaussian_patch(shape, centre, sigma, amp):
    """Generate a ±1σ Gaussian bump of size shape at centre."""
    h, w = shape
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    sy, sx = sigma
    patch = np.exp(-(((yy - centre[0])**2)/(2*sy**2) + ((xx - centre[1])**2)/(2*sx**2)))
    patch *= amp / patch.max()
    patch *= random.choice([-1, 1])
    return patch


def apply_patch(field: torch.Tensor, patch: np.ndarray) -> torch.Tensor:
    """Apply 2D patch to all frames and channels of field."""
    p = torch.from_numpy(patch).to(field.device, dtype=field.dtype)
    return field + p

# ---------------------------------------------------------------------------
# 1) GPU‐only Gaussian patch generator (always returns a torch.Tensor on DEVICE)
# ---------------------------------------------------------------------------
def gaussian_patch_gpu(shape, centre, sigma, amp, device):
    """
    Generate a ±1σ Gaussian bump of size `shape` at `centre`,
    entirely on `device` (GPU).
    """
    H, W = shape
    sy, sx = sigma

    # build coordinate grids on GPU:
    yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W)

    # Gaussian kernel formula:
    patch = torch.exp(-(((yy - centre[0])**2) / (2 * sy**2)
                       + ((xx - centre[1])**2) / (2 * sx**2)))
    patch = patch * (amp / patch.max())

    # randomly flip sign ±1
    sign = 1.0 if random.random() < 0.5 else -1.0
    patch = patch * sign

    return patch  # shape (H, W), dtype=float32, on DEVICE


# ---------------------------------------------------------------------------
# 2) Broadcaster: apply that patch to a (B, T, H, W) torch.Tensor
# ---------------------------------------------------------------------------
def apply_patch_gpu(field: torch.Tensor, patch) -> torch.Tensor:
    """
    field: torch.Tensor of shape (B, T, H, W)
    patch: either a numpy array or torch.Tensor of shape (H, W)

    Returns field + patch broadcast to (B, T, H, W), all on GPU.
    """
    # if user accidentally passed a NumPy array, convert it:
    if not isinstance(patch, torch.Tensor):
        patch = torch.as_tensor(patch, dtype=field.dtype, device=field.device)

    # now patch is (H, W); unsqueeze to (1, 1, H, W) and rely on broadcasting
    return field + patch.unsqueeze(0).unsqueeze(0)


# -----------------------------------------------------------------------------
# ENGINE
# -----------------------------------------------------------------------------
class SpatialPerturbationEngine:
    def __init__(self, model: torch.nn.Module, loader: DataLoader, n_iter: int = 50000, conv_tol: float = 0.005):
        self.model  = model.to(DEVICE).eval()
        self.loader = loader
        self.n_iter_target = n_iter       # total Monte‑Carlo shots
        self.conv_tol = conv_tol
        sample = next(iter(loader))['ppt']  # (1, T, H, W)
        _, _, self.H, self.W = sample.shape
        self.running = None
        self.count   = 0

    def _perturb_once(self, batch):
        # choose a random pixel center
        centre = (random.randrange(self.H), random.randrange(self.W))

        # 1) generate a single patch **on the GPU**
        patch = gaussian_patch_gpu(
            shape=(self.H, self.W),
            centre=centre,
            sigma=SIGMA_PIX,
            amp=AMPLITUDE,
            device=DEVICE
        )

        # 2) bring your input fields to GPU
        ppt  = batch['ppt'].to(DEVICE)   # (B, T, H, W)
        tmin = batch['tmin'].to(DEVICE)
        tmax = batch['tmax'].to(DEVICE)

        # 3) baseline outputs (B, n_gauges)
        with torch.no_grad():
            base = self.model(ppt, tmin, tmax).cpu().numpy()[0]

        # 4) apply the patch across **all** time‐steps & samples
        ppt_p  = apply_patch_gpu(ppt,  patch)
        tmin_p = apply_patch_gpu(tmin, patch)
        tmax_p = apply_patch_gpu(tmax, patch)

        # 5) perturbed outputs
        with torch.no_grad():
            pert = self.model(ppt_p, tmin_p, tmax_p).cpu().numpy()[0]

        # 6) compute your per‐gauge sensitivity
        delta = np.abs(base - pert)           # (n_gauges,)
        sens  = delta[:, None, None] * patch.cpu().numpy()  # back to NumPy for accumulation
        return sens



    def run(self) -> np.ndarray:
        """Run Monte–Carlo perturbations across all days, then check convergence."""
        # Prime with first batch to get shapes & station count
        first = next(iter(self.loader))
        n_gauges = self.model(
            first['ppt'].to(DEVICE),
            first['tmin'].to(DEVICE),
            first['tmax'].to(DEVICE)
        ).shape[-1]

        self.running = np.zeros((n_gauges, self.H, self.W), dtype=np.float32)
        prev = np.zeros_like(self.running)
        self.count = 0
        converged = False
        # outer loop with a named tqdm so we can update its postfix
        outer = tqdm(
            range(self.n_iter_target),
            desc="Perturb iters",
            unit="iter"
        )

        for iteration in outer:
            # inner loop over batches
            for batch in tqdm(
                self.loader,
                desc=f"  Batches (it {iteration})",
                leave=False,
                unit="batch"
            ):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                sens = self._perturb_once(batch)   # (n_gauges, H, W)
                self.count += 1
                # online mean update
                self.running += (sens - self.running) / self.count
                # new: always positive denominator
                num = np.abs(self.running - prev).mean()
                den = np.abs(prev).mean() + 1e-9
                rel = num / den
                # push rel into the outer bar
                outer.set_postfix(rel=f"{rel:.4f}")
                
                # after a full pass through all days, test convergence every 10 iters
                tqdm.write(f"[Iter {iteration}] rel change = {rel:.4f}")
                if rel < self.conv_tol:
                    tqdm.write(
                        f"Converged after {iteration} iterations"
                        f" ({self.count} total perturbations)."
                    )
                    converged = True
                    break
                prev[:] = self.running  # snapshot for next check
            if converged:
                break
        return self.running


# -----------------------------------------------------------------------------
# SAVE MAPS
# -----------------------------------------------------------------------------

def save_sensitivity_maps(sens_maps: np.ndarray, gauges: pd.DataFrame, output_dir: Path):
    """Save each station map (normalized individually) clipped to the Michigan boundary 
    with a station marker using the RdBu_r color scheme.
    Each map is normalized individually to its maximum absolute value.
    """
    output_dir.mkdir(exist_ok=True)
    # Read boundary
    gdf = gpd.read_file(SHAPE_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    minx, miny, maxx, maxy = gdf.total_bounds
    H, W = sens_maps.shape[1:]
    # Build transform
    xres = (maxx - minx) / W
    yres = (maxy - miny) / H
    transform = Affine(xres, 0, minx, 0, -yres, maxy)
    # Rasterize (mask the area outside the boundary)
    mask = geometry_mask(((geom, 1) for geom in gdf.geometry),
                         out_shape=(H, W), transform=transform, invert=True)
    
    # Iterate through each station
    for i, site in enumerate(gauges['station_id']):
        sens_map = sens_maps[i]
        # Clip the sensitivity map using the mask
        clipped = np.where(mask, sens_map, np.nan)
        # Normalize the map individually using its maximum absolute value.
        max_val = np.nanmax(np.abs(clipped))
        if max_val != 0:
            normed = clipped / max_val
        else:
            normed = clipped  # if the map is zero everywhere, leave it unchanged
        
        lat_s = gauges['lat'].iloc[i]
        lon_s = gauges['lon'].iloc[i]
        # Convert station lon/lat to pixel coordinates
        inv = ~transform
        col, row = map(int, map(round, inv * (lon_s, lat_s)))
        
        # Plot using the RdBu_r color scheme with fixed limits (-1 to 1)
        plt.figure(figsize=(6, 6))
        im = plt.imshow(normed, cmap='RdBu_r', vmin=-1, vmax=1, origin='upper')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.scatter(col, row, marker='X', s=100, edgecolor='white', facecolor='black')
        plt.title(f'Sensitivity - {site}')
        plt.axis('off')
        plt.savefig(output_dir / f'sensitivity_{site}.png', dpi=200, bbox_inches='tight')
        plt.close()

# -----------------------------------------------------------------------------
# CLUSTER-LEVEL EVALUATION
# -----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from affine import Affine
from scipy.stats import ks_2samp
from rasterio.features import geometry_mask
from pathlib import Path
from typing import Optional, Tuple

def cluster_evaluation(
    sens_maps: np.ndarray,
    gauges_df: pd.DataFrame,
    shape_path: Path,
    output_dir: Path,
    grid_bounds: Optional[tuple[float, float, float, float]] = None
):
    """
    1) Normalises each station map to ±1 before averaging.
    2) Computes KS–D and footprint–A per cluster using watershed polygons only.
    3) Saves a split-violin plot (Fig. 8 style) of inside vs outside sensitivity.
    Returns a DataFrame with columns ['cluster','D','A'].
    """
    # ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read & project watershed polygons
    basins = (
        gpd.read_file(shape_path)
        .set_crs("EPSG:4326", allow_override=True)
        .to_crs("EPSG:4326")
    )

    # 2. Build gauge GeoDataFrame and spatially join to get HUC8 per gauge
    gauges_gdf = gpd.GeoDataFrame(
        gauges_df.copy(),
        geometry=gpd.points_from_xy(gauges_df.lon, gauges_df.lat),
        crs="EPSG:4326"
    )
    gauges_w = gpd.sjoin(
        gauges_gdf, 
        basins[["HUC8", "geometry"]],
        how="left",
        predicate="within"
    )

    # 3. Define raster affine
    if grid_bounds is None:
        minx, miny, maxx, maxy = basins.total_bounds
    else:
        minx, miny, maxx, maxy = grid_bounds
    H, W = sens_maps.shape[1], sens_maps.shape[2]
    xres, yres = (maxx - minx)/W, (maxy - miny)/H
    transform = Affine(xres, 0, minx, 0, -yres, maxy)

    # 4. Prepare accumulators
    records = []
    fig8_rows = []
    cluster_ids = sorted(gauges_df["cluster"].unique())

    # 5. Loop over clusters
    for c in cluster_ids:
        # indices of gauges in this cluster
        idx = gauges_w.index[gauges_w["cluster"] == c]
        if len(idx) == 0:
            continue

        # union all HUC8 polygons for this cluster
        hucs = gauges_w.loc[idx, "HUC8"].dropna().unique().tolist()
        polys = basins.loc[basins["HUC8"].isin(hucs), "geometry"]
        cluster_ws = polys.unary_union if not polys.empty else basins.unary_union

        # create inside‐watershed mask
        inside_mask = geometry_mask(
            [(cluster_ws, 1)],
            out_shape=(H, W),
            transform=transform,
            invert=True
        )

        # 5a. Normalize each station map and compute its footprint-A
        stack = []
        A_vals = []
        for i in idx:
            S_raw = sens_maps[i]
            if not np.isfinite(S_raw).any():
                continue

            max_abs = np.nanmax(np.abs(S_raw))
            if max_abs == 0 or np.isnan(max_abs):
                continue

            # normalize to ±1
            S = S_raw / max_abs
            stack.append(S)

            finite = np.isfinite(S)
            # footprint‐A: fraction of pixels whose |S| > 0.5
            A_vals.append((np.abs(S[finite]) > 0.5).sum() / finite.sum())

        if not stack:
            D_val, A_val, S_mean = np.nan, np.nan, np.full((H, W), np.nan)
        else:
            # cluster‐mean map on normalized maps
            S_mean = np.nanmean(stack, axis=0)

            # split into inside / outside samples
            inside  = S_mean[inside_mask].ravel()
            outside = S_mean[~inside_mask].ravel()
            inside  = inside[np.isfinite(inside)]
            outside = outside[np.isfinite(outside)]

            # KS–D
            if inside.size >= 2 and outside.size >= 2:
                D_val = ks_2samp(inside, outside, mode="asymp").statistic
            else:
                D_val = np.nan

            # median footprint-A
            A_val = float(np.nanmedian(A_vals))

            # collect for violin plot
            fig8_rows.append({"cluster": c, "group": "Inside",  "sens": inside})
            fig8_rows.append({"cluster": c, "group": "Outside", "sens": outside})

        records.append({"cluster": int(c), "D": D_val, "A": A_val})

    # Create lookup for D‐values
    cluster_D = {r["cluster"]: r["D"] for r in records}

    # 6. Build DataFrame for violin
    fig8_df = (
        pd.DataFrame(fig8_rows)
          .explode("sens", ignore_index=True)
          .dropna(subset=["sens"])
    )

    # 7. Plot split‐violin (Fig. 8)
    palette = {"Inside": "#C95F3F", "Outside": "#48B2C0"}
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.violinplot(
        data=fig8_df,
        x="cluster", y="sens",
        hue="group", split=True, inner=None,
        palette=palette, linewidth=0, ax=ax
    )
    # annotate D
    for i, c in enumerate(cluster_ids):
        if np.isfinite(cluster_D[c]):
            txt = f"$\mathit{{D}}={cluster_D[c]:.2f}$"
        else:
            txt = r"$\mathit{D}=	ext{—}$"
        ax.text(i, ax.get_ylim()[1] * 0.95, txt,
                ha="center", va="top", fontsize=11, fontweight="bold")

    ax.set_xlabel("")
    ax.set_ylabel("Sensitivity", fontsize=13)
    ax.set_title("Sensitivity distributions inside vs outside cluster watersheds", fontsize=15, pad=12)
    ax.legend(title="", loc="lower center", frameon=True)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(output_dir / "figure8b_violin.png", dpi=300)
    plt.close(fig)

    # 8. Return summary metrics
    return pd.DataFrame(records)



# -------------------------------------------------------------------------
# PLOT CLUSTER-LEVEL MAPS
# -------------------------------------------------------------------------
def save_cluster_maps(sens_maps, clusters_df, shape_path, output_dir):
    """
    Average per-cluster sensitivity maps, clip to watershed,
    overlay station locations from clusters_df, and save per-cluster.
    clusters_df must contain 'site_no', 'cluster_id', 'dec_lat_va', 'dec_long_v'.
    """
    output_dir.mkdir(exist_ok=True)
    # load and prepare watershed mask
    gdf = gpd.read_file(shape_path)
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    minx, miny, maxx, maxy = gdf.total_bounds
    H, W = sens_maps.shape[1:]
    xres = (maxx - minx)/W
    yres = (maxy - miny)/H
    transform = Affine(xres,0,minx,0,-yres,maxy)
    mask = geometry_mask(((geom,1) for geom in gdf.geometry), out_shape=(H,W), transform=transform, invert=True)

    # iterate clusters
    for c in sorted(clusters_df['cluster'].unique()):
        idx = clusters_df.index[clusters_df['cluster'] == c].tolist()
        S = sens_maps[idx].mean(axis=0)
        clipped = np.where(mask, S, np.nan)
        plt.figure(figsize=(6,6))
        im = plt.imshow(clipped, cmap='hot', origin='upper')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        # plot stations from clusters_df
        for _, row in clusters_df[clusters_df['cluster'] == c].iterrows():
            lon = row['lon']; lat = row['lat']
            col_px, row_px = map(int, map(round, (~transform) * (lon, lat)))
            plt.scatter(col_px, row_px, marker='o', s=50, edgecolor='white', facecolor='black')
        plt.title(f'Cluster {c} Mean Sensitivity')
        plt.axis('off')
        plt.savefig(output_dir / f'cluster_{c}_sensitivity.png', dpi=200, bbox_inches='tight')
        plt.close()



from pathlib import Path
from typing import Optional, Tuple

def plot_cluster_watershed_bedrock_grid(
    sens_maps: np.ndarray,
    clusters_df: pd.DataFrame,
    shape_path: Path,
    bedrock_path: Path,
    output_path: Path,
    cmap: str = "RdBu_r",
) -> pd.DataFrame:
    """
    Plot a 2×2 grid of cluster‑mean sensitivity maps and return cluster‑level
    KS‑D and footprint‑A.  Also produces an extra “Fig‑8” violin plot that
    contrasts the sensitivity distributions inside vs outside each cluster
    watershed (saved as *figure8_violin.png* next to `output_path`).

    ▸ Each station map is normalised to ±1 before averaging.
    ▸ 'Inside' mask  = watershed ∪ full bedrock.
    ▸ 'Far' mask     = pixels ≥ 50 raster cells outside that union.
    ▸ D is computed once per cluster on the mean map.
    ▸ A is computed per station, then the cluster median is reported.
    """
    # ------------------------------------------------------------------
    # 0. Imports (kept inside for portability)
    # ------------------------------------------------------------------
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import seaborn as sns                      # ← NEW
    from rasterio.features import geometry_mask
    from affine import Affine
    from scipy.stats import ks_2samp
    import numpy as np
    import pandas as pd
    output_path = Path(output_path)
    # ------------------------------------------------------------------
    # 1. Read layers
    # ------------------------------------------------------------------
    ws = gpd.read_file(shape_path).to_crs("EPSG:4326")
    br = gpd.read_file(bedrock_path).to_crs("EPSG:4326")
    full_br = br.geometry.union_all()

    # ------------------------------------------------------------------
    # 2. Raster geometry
    # ------------------------------------------------------------------
    cluster_ids = sorted(clusters_df.cluster.unique())
    if len(cluster_ids) != 4:
        raise ValueError(f"Expected exactly 4 clusters, got {len(cluster_ids)}")

    H, W = sens_maps.shape[1:]
    full_union = ws.geometry.union_all().union(full_br)
    minx, miny, maxx, maxy = full_union.bounds
    xres = (maxx - minx) / W
    yres = (maxy - miny) / H
    transform = Affine(xres, 0, minx, 0, -yres, maxy)
    extent = (minx, maxx, miny, maxy)

    # ------------------------------------------------------------------
    # 3. Containers
    # ------------------------------------------------------------------
    cluster_D, cluster_A, mean_maps = {}, {}, {}
    fig8_rows = []                                # ← NEW (collect samples)

    # ------------------------------------------------------------------
    # 4. Loop over clusters
    # ------------------------------------------------------------------
    for c in cluster_ids:
        subs = clusters_df[clusters_df.cluster == c]

        # 4a. Watershed ∪ bedrock geometry
        pts        = gpd.points_from_xy(subs.lon, subs.lat, crs="EPSG:4326")
        mask_ws    = ws.geometry.apply(lambda poly: any(pt.within(poly) for pt in pts))
        cluster_ws = ws.geometry[mask_ws].union_all()
        region_union = cluster_ws.union(full_br)

        # 4b. Masks: inside / halo / far
        inside_mask = geometry_mask(
            [(region_union, 1)],
            out_shape=(H, W),
            transform=transform,
            invert=True,
        )

        buffer_deg = 50 * max(xres, yres)         # 50‑pixel buffer
        region_buf = region_union.buffer(buffer_deg)

        halo_mask = geometry_mask(
            [(region_buf, 1)],
            out_shape=(H, W),
            transform=transform,
            invert=True,
        )
        far_mask = ~halo_mask

        # 4c. Normalise maps; accumulate A
        stack, A_vals = [], []
        for i in subs.index:
            S_raw = sens_maps[i]
            if not np.isfinite(S_raw).any():
                continue
            max_abs = np.nanmax(np.abs(S_raw))
            if max_abs == 0 or np.isnan(max_abs):
                continue

            S = S_raw / max_abs
            stack.append(S)

            finite = np.isfinite(S)
            A = np.count_nonzero(np.abs(S[finite]) > 0.5) / finite.sum()
            A_vals.append(A)

        if len(stack) == 0:
            cluster_D[c] = np.nan
            cluster_A[c] = np.nan
            mean_maps[c] = np.full((H, W), np.nan)
            continue

        # 4d. Cluster‑mean map & KS‑D
        S_mean = np.nanmean(stack, axis=0)
        mean_maps[c] = S_mean

        inside  = S_mean[inside_mask].ravel()
        outside = S_mean[far_mask].ravel()
        inside  = inside[np.isfinite(inside)]
        outside = outside[np.isfinite(outside)]

        if inside.size >= 2 and outside.size >= 2:
            D_val = ks_2samp(inside, outside, mode="asymp").statistic
        else:
            D_val = np.nan

        cluster_D[c] = D_val
        cluster_A[c] = np.nanmedian(A_vals)

        # 4e. Collect samples for Fig‑8                               ← NEW
        fig8_rows.append({"cluster": str(c), "group": "Inside",  "sens": inside,  "D": D_val})
        fig8_rows.append({"cluster": str(c), "group": "Outside", "sens": outside, "D": D_val})

    # ------------------------------------------------------------------
    # 5. 2×2 Map figure (unchanged)
    # ------------------------------------------------------------------
    vlim = max(np.nanmax(np.abs(m)) for m in mean_maps.values())
    data_ratio = (maxy - miny) / (maxx - minx)
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 12 * data_ratio),
        subplot_kw={"xticks": [], "yticks": [], "aspect": data_ratio},
        constrained_layout=True,
    )
    axes = axes.flatten()
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, c, lbl in zip(axes, cluster_ids, labels):
        S = mean_maps[c]
        im = ax.imshow(S, cmap=cmap, vmin=-vlim, vmax=vlim,
                       extent=extent, origin="upper")

        ws.boundary.plot(ax=ax, edgecolor="gray", linestyle="--", linewidth=1)
        ax.scatter(
            clusters_df.loc[clusters_df.cluster == c, "lon"],
            clusters_df.loc[clusters_df.cluster == c, "lat"],
            facecolors="none", edgecolors="black", s=40, linewidth=1,
        )
        # outline: watershed & bedrock for *this* cluster
        pts   = gpd.points_from_xy(
                    clusters_df.loc[clusters_df.cluster == c, "lon"],
                    clusters_df.loc[clusters_df.cluster == c, "lat"],
                    crs="EPSG:4326")
        mask_ws = ws.geometry.apply(lambda poly: any(pt.within(poly) for pt in pts))
        gpd.GeoSeries(ws.geometry[mask_ws].union_all()).boundary.plot(
            ax=ax, edgecolor="blue", linewidth=1.5)
        gpd.GeoSeries(full_br).boundary.plot(
            ax=ax, edgecolor="green", linewidth=1.5)

        ax.set_title(f"Cluster {c}", fontsize=14, pad=6)
        ax.text(0.02, 0.95, lbl, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")
        if ax is axes[0]:
            ax.legend(["Full watershed", "Cluster WS", "Full bedrock"],
                      loc="lower left")

    cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                        fraction=0.04, pad=0.02)
    cbar.set_label("Sensitivity $S(x,y)$", fontsize=14)
    fig.savefig(output_path / "figure7_clusters.png", dpi=1000)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. Fig‑8 split‑violin plot                                   ← NEW
    # ------------------------------------------------------------------
    fig8_df = (pd.DataFrame(fig8_rows)
               .explode("sens", ignore_index=True)
               .dropna(subset=["sens"]))

    palette_light  = {"Inside": "#C95F3F", "Outside": "#48B2C0"}

    fig8, ax8 = plt.subplots(figsize=(9, 7))
    sns.violinplot(
        data=fig8_df, x="cluster", y="sens",
        hue="group", split=True, inner=None,  # no dashes at all
        palette=palette_light, linewidth=0, ax=ax8,
    )

    # annotate D values
    for i, c in enumerate(cluster_ids):
        if np.isfinite(cluster_D[c]):
            txt = f"$\mathit{{D}}={cluster_D[c]:.2f}$"
        else:
            txt = r"$\mathit{D}=	ext{—}$"
        ax8.text(i, ax8.get_ylim()[1] * 0.95, txt,
                 ha="center", va="top", fontsize=11, fontweight="bold")

    ax8.set_xlabel("")
    ax8.set_ylabel("Sensitivity", fontsize=13)
    ax8.set_title("Sensitivity distributions inside vs outside cluster watersheds",
                  fontsize=15, pad=12)
    ax8.legend(title="", loc="lower center", frameon=True)
    sns.despine(fig8)
    fig8.tight_layout()
    fig8.savefig(output_path / "figure8_violin.png", dpi=1000)
    plt.close(fig8)

    # ------------------------------------------------------------------
    # 7. Return metrics
    # ------------------------------------------------------------------
    return pd.DataFrame(
        {"cluster": cluster_ids,
         "D": [cluster_D[c] for c in cluster_ids],
         "A": [cluster_A[c] for c in cluster_ids]}
    )

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    # dataset and loader
    ds     = HDF5Dataset(H5_FILE, ['ppt','tmin','tmax'], LABELS_CSV, 2000, 2009)
    num_train = int(0.8 * len(ds))
    num_val = int(0.1 * len(ds))
    num_test = len(ds) - num_train - num_val
    # Randomly split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(ds, [num_train, num_val, num_test])
    loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)

    # if we've run before, just reload the numpy array
    if CACHE_PATH.exists():
        print(f"Loading cached sensitivity maps from {CACHE_PATH!r}")
        sens_maps = np.load(CACHE_PATH)
    else:
        # model
        model = CNN_LSTM()
        ckpt  = torch.load(CHECKPOINT, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        
        model.to(DEVICE)
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model.eval()

        # run engine once
        engine    = SpatialPerturbationEngine(model, loader)
        sens_maps = engine.run()

        # save to disk for next time
        np.save(CACHE_PATH, sens_maps)
        print(f"Saved sensitivity maps to {CACHE_PATH!r}")

    return sens_maps

if __name__ == '__main__':
    sens_maps = main()
    gauges = load_gauge_metadata(USGS_METADATA)
    save_sensitivity_maps(sens_maps, gauges, Path('sigma_100/sensitivity_maps'))
    # # load gauges and clusters
    gauges = pd.read_csv(USGS_METADATA,dtype={'site_no':str})

    metrics = cluster_evaluation(sens_maps, gauges, SHAPE_PATH, 'sigma_100/cluster_maps')
    print('Watershed based', metrics)
    
    # 4) multi‐panel figure
    metrics = plot_cluster_watershed_bedrock_grid(
        sens_maps=sens_maps,
        clusters_df=gauges,
        shape_path=SHAPE_PATH,
        bedrock_path=BEDROCK_PATH,
        output_path='sigma_100/cluster_maps',
    )
    print('Watershed + Bedrock based', metrics)
    # call cluster map plotting
    save_cluster_maps(sens_maps, gauges, SHAPE_PATH, Path('sigma_100/cluster_maps'))
    
    
    
    

