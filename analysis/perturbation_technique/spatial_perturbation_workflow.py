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
from typing import Optional, Tuple
# allow project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_loader import HDF5Dataset
from model import CNN_LSTM
from shapely.errors import ShapelyDeprecationWarning
import warnings
from shapely.geometry import Point
from scipy.stats import ks_2samp

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
# after loading CONF:
CACHE_PATH = Path(CONF.get("sensitivity_cache", "sens_maps.npy"))
# Perturbation params
SIGMA_PIX  = tuple(CONF.get('sigma_pix', CONF.get('sigma_pixels', [75, 75])))
AMPLITUDE  = float(CONF.get('amplitude', 1.0))
BATCH_SIZE = int(1)
NUM_WORKERS= int(CONF.get('num_workers', 4))
# Device selection
gpu_list = CONF.get('gpu', [])
if torch.cuda.is_available() and gpu_list:
    idx = int(gpu_list[0])
    DEVICE = torch.device(f'cuda:{idx}') if idx < torch.cuda.device_count() else torch.device('cpu')
else:
    DEVICE = torch.device('cpu')

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
        centre = (random.randrange(self.H), random.randrange(self.W))
        patch  = gaussian_patch((self.H, self.W), centre, SIGMA_PIX, AMPLITUDE)
        # baseline (detached)
        with torch.no_grad():
            base = self.model(batch['ppt'], batch['tmin'], batch['tmax']).detach().cpu().numpy()[0]
        # perturb
        ppt_p  = apply_patch(batch['ppt'], patch)
        tmin_p = apply_patch(batch['tmin'], patch)
        tmax_p = apply_patch(batch['tmax'], patch)
        with torch.no_grad():
            pert = self.model(ppt_p, tmin_p, tmax_p).detach().cpu().numpy()[0]
        delta = np.abs(base - pert)           # (n_gauges,)
        sens  = delta[:, None, None] * patch  # (n_gauges, H, W)
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
            if iteration and iteration % 10 == 0:
                tqdm.write(f"[Iter {iteration}] rel change = {rel:.4f}")
                if rel < self.conv_tol:
                    tqdm.write(
                        f"Converged after {iteration} iterations"
                        f" ({self.count} total perturbations)."
                    )
                    break
                prev[:] = self.running  # snapshot for next check

        return self.running


# -----------------------------------------------------------------------------
# SAVE MAPS
# -----------------------------------------------------------------------------

def save_sensitivity_maps(sens_maps: np.ndarray, gauges: pd.DataFrame, output_dir: Path):
    """Save each station map clipped to Michigan boundary and overlay station marker."""
    output_dir.mkdir(exist_ok=True)
    # read boundary
    gdf = gpd.read_file(SHAPE_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    minx, miny, maxx, maxy = gdf.total_bounds
    H, W = sens_maps.shape[1:]
    # build transform
    xres = (maxx - minx) / W
    yres = (maxy - miny) / H
    transform = Affine(xres, 0, minx, 0, -yres, maxy)
    # rasterize
    mask = geometry_mask(((geom,1) for geom in gdf.geometry), out_shape=(H,W), transform=transform, invert=True)
    # iterate stations
    for i, site in enumerate(gauges['station_id']):
        sens_map = sens_maps[i]
        clipped = np.where(mask, sens_map, np.nan)
        lat_s = gauges['lat'].iloc[i]
        lon_s = gauges['lon'].iloc[i]
        # pixel coords
        inv = ~transform
        col, row = map(int, map(round, inv * (lon_s, lat_s)))
        # plot
        plt.figure(figsize=(6,6))
        im = plt.imshow(clipped, cmap='hot', origin='upper')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.scatter(col, row, marker='X', s=100, edgecolor='white', facecolor='black')
        plt.title(f'Sensitivity - {site}')
        plt.axis('off')
        plt.savefig(output_dir / f'sensitivity_{site}.png', dpi=200, bbox_inches='tight')
        plt.close()

# -----------------------------------------------------------------------------
# CLUSTER-LEVEL EVALUATION
# -----------------------------------------------------------------------------
def cluster_evaluation(
    sens_maps: np.ndarray,
    gauges_df: pd.DataFrame,
    shape_path: Path,
    grid_bounds: Optional[Tuple[float, float, float, float]] = None
) -> pd.DataFrame:
    """
    Compute KS-D and footprint-A for each cluster using true watershed polygons.

    Parameters
    ----------
    sens_maps   : (n_stations, H, W) array of sensitivity maps.
    gauges_df   : DataFrame with ['station_id','cluster','lat','lon', …].
    shape_path  : Path to Watershed_Boundary_Intersect_Michigan.shp.
    grid_bounds : (minx, miny, maxx, maxy) of the raster grid in lon/lat;
                  if None, uses the shapefile’s total bounds.

    Returns
    -------
    DataFrame with columns ['cluster','D','A'].
    """
    # 1. Load watershed polygons
    basins = gpd.read_file(shape_path)
    if basins.crs is None:
        basins = basins.set_crs("EPSG:4326")
    basins = basins.to_crs("EPSG:4326")

    # 2. Tag each gauge with its containing HUC8
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

    # 3. Build raster affine transform
    if grid_bounds is None:
        minx, miny, maxx, maxy = basins.total_bounds
    else:
        minx, miny, maxx, maxy = grid_bounds

    H, W = sens_maps.shape[1], sens_maps.shape[2]
    xres = (maxx - minx) / W
    yres = (maxy - miny) / H
    transform = Affine(xres, 0, minx, 0, -yres, maxy)

    # 4. Compute D and A per cluster
    records = []
    for c in sorted(gauges_df["cluster"].unique()):
        idx = gauges_w.index[gauges_w["cluster"] == c]
        if len(idx) == 0:
            continue

        # union of HUC8 polygons for this cluster’s gauges
        hucs = gauges_w.loc[idx, "HUC8"].dropna().unique().tolist()
        polys = basins.loc[basins["HUC8"].isin(hucs), "geometry"]
        if polys.empty:
            polys = basins.geometry  # fallback

        mask = geometry_mask(
            ((geom, 1) for geom in polys),
            out_shape=(H, W),
            transform=transform,
            invert=True
        )

        S = sens_maps[idx].mean(axis=0)
        inside  = S[mask]
        outside = S[~mask]

        if inside.size == 0 or outside.size == 0:
            D = np.nan
        else:
            D = ks_2samp(inside, outside, mode="asymp").statistic

        thresh = 0.5 * np.nanmax(np.abs(S))
        A = float((np.abs(S) > thresh).sum() / S.size)

        records.append({"cluster": int(c), "D": float(D), "A": A})

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
            
            
# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    # dataset and loader
    ds     = HDF5Dataset(H5_FILE, ['ppt','tmin','tmax'], LABELS_CSV, 2009, 2009)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # if we've run before, just reload the numpy array
    if CACHE_PATH.exists():
        print(f"Loading cached sensitivity maps from {CACHE_PATH!r}")
        sens_maps = np.load(CACHE_PATH)
    else:
        # model
        model = CNN_LSTM()
        ckpt  = torch.load(CHECKPOINT, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
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
    save_sensitivity_maps(sens_maps, gauges, Path('sensitivity_maps'))
    # load gauges and clusters
    # gauges = pd.read_csv(USGS_METADATA,dtype={'site_no':str})
    # clusters_df = pd.read_csv(CLUSTERS_CSV,dtype={'station_id':str,'cluster':int})
    # cluster evaluation
    metrics = cluster_evaluation(sens_maps, gauges, SHAPE_PATH)
    print(metrics)
    
    # call cluster map plotting
    save_cluster_maps(sens_maps, gauges, SHAPE_PATH, Path('cluster_maps'))