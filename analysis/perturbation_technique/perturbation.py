#!/usr/bin/env python3
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Union, Tuple
from scipy.ndimage import gaussian_filter
import fiona
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine
# Ensure project root is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_loader import HDF5Dataset
from model import CNN_LSTM

import pandas as pd
import geopandas as gpd

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def load_yaml_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

# Load YAML configuration globally
config = load_yaml_config('../../config/config.yaml')


def create_gaussian_mask(
    shape: Tuple[int,int],
    center: Tuple[int,int],
    sigma: Union[float, Tuple[float,float]],
    amplitude: float = 1.0
) -> np.ndarray:
    """Generate a 2D Gaussian mask."""
    H, W = shape
    # Unpack sigma
    try:
        sigma_y, sigma_x = sigma  # type: ignore
    except Exception:
        sigma_y = sigma_x = float(sigma)  # type: ignore
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    gauss = np.exp(
        -(((y - center[0])**2)/(2*sigma_y**2)
          +((x - center[1])**2)/(2*sigma_x**2))
    )
    gauss *= amplitude/gauss.max()
    phi = np.random.choice([-1,1])
    return gauss * phi

def create_gaussian_patch_mask(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    patch_size: int = 11,
    sigma: Union[float, Tuple[float, float]] = 1.5,
    amplitude: float = 1.0
) -> np.ndarray:
    """
    Create a small square patch centered at 'center', then smooth with a Gaussian filter
    allowing different std dev in y and x directions.

    Parameters
    ----------
    shape : (H, W)
    center: (y, x)
    patch_size: side length of square patch in pixels
    sigma: float or (sigma_y, sigma_x) standard deviations for Gaussian filter
    amplitude: peak perturbation magnitude

    Returns
    -------
    mask : np.ndarray of shape (H, W)
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=float)
    half = patch_size // 2
    y0, x0 = center
    # define patch region
    y1 = max(0, y0 - half)
    y2 = min(H, y0 + half + 1)
    x1 = max(0, x0 - half)
    x2 = min(W, x0 + half + 1)
    mask[y1:y2, x1:x2] = 1.0

    # unpack anisotropic sigma
    try:
        sigma_y, sigma_x = sigma  # type: ignore
    except Exception:
        sigma_y = sigma_x = float(sigma)  # type: ignore

    # apply Gaussian smoothing with separate y/x std devs
    mask = gaussian_filter(mask, sigma=(sigma_y, sigma_x))

    # normalize to amplitude
    max_val = mask.max()
    if max_val > 0:
        mask = mask * (amplitude / max_val)

    # random sign flip
    # mask *= np.random.choice([-1, 1])
    return mask

def apply_gaussian_perturbation(
    data: np.ndarray,
    center: Tuple[int, int],
    sigma: Union[float, Tuple[float, float]],
    patch_size: int = 0,
    amplitude: float = 1.0
) -> np.ndarray:
    """Apply a 2D Gaussian perturbation mask to 4D or 5D climate data."""
    mask = create_gaussian_patch_mask(data.shape[-2:], center, patch_size, sigma, amplitude)
    # scale mask by data std to reflect one std deviation in normalized units
    data_std = data.std()
    scaled_mask = mask * data_std
    if data.ndim == 4:
        # (batch, time, H, W)
        return data + scaled_mask[None, None, :, :], mask
    elif data.ndim == 5:
        # (batch, time, channels, H, W)
        return data + scaled_mask[None, None, None, :, :], mask
    else:
        raise ValueError("Data array must be 4D or 5D")


def run_inference(model, data_loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in data_loader:
            ppt = batch['ppt'].to(device)
            tmin = batch['tmin'].to(device)
            tmax = batch['tmax'].to(device)
            out = model(ppt, tmin, tmax)
            outputs.append(out.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def calculate_sensitivity(orig: np.ndarray, pert: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pert - orig), axis=0)


def plot_four_panel(
    mask: np.ndarray,
    original_field: np.ndarray,
    perturbed_field: np.ndarray,
    diff_map: np.ndarray,
    save_path: str
):

    """Plot (a) mask, (b) original, (c) perturbed, (d) diff_map."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    im0 = axes[0,0].imshow(mask, cmap='viridis')
    axes[0,0].set_title('(a) Gaussian perturbation')
    fig.colorbar(im0, ax=axes[0,0])

    im1 = axes[0,1].imshow(original_field, cmap='viridis')
    axes[0,1].set_title('(b) Original Tmax')
    fig.colorbar(im1, ax=axes[0,1])

    im2 = axes[1,0].imshow(perturbed_field, cmap='viridis')
    axes[1,0].set_title('(c) Perturbed Tmax')
    fig.colorbar(im2, ax=axes[1,0])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved 3-panel plot: {save_path}")

def plot_combined_panel(mask: np.ndarray,
                        original_field: np.ndarray,
                        perturbed_field: np.ndarray,
                        sens: np.ndarray,
                        csv_path: str,
                        site_numbers: list,
                        shp_path: str,
                        save_path: str):
    """
    Create a 2x2 panel plot:
      (a) Gaussian perturbation mask,
      (b) Original Tmax field,
      (c) Perturbed Tmax field,
      (d) Sensitivity map overlayed on the shapefile boundary with gauge points.

    Parameters
    ----------
    mask : np.ndarray
        The perturbation mask (2D array).
    original_field : np.ndarray
        The original field (2D array).
    perturbed_field : np.ndarray
        The perturbed field (2D array).
    sens : np.ndarray
        1D array of sensitivity values (shape: (61,)).
    csv_path : str
        Path to the CSV file with USGS gauge locations.
    site_numbers : list
        List of 61 gauge site numbers (integers).
    shp_path : str
        Path to the shapefile.
    save_path : str
        File path to save the resulting plot.
    """
    import matplotlib.pyplot as plt

    # Prepare gauge locations from CSV for panel (d)
    import pandas as pd
    import geopandas as gpd
    df = pd.read_csv(csv_path, dtype={'site_no': str})
    df['site_no_int'] = df['site_no'].astype(int)
    df_filtered = df[df['site_no_int'].isin(site_numbers)]
    lats = []
    lons = []
    for sn in site_numbers:
        row = df_filtered[df_filtered['site_no_int'] == sn]
        if not row.empty:
            lats.append(float(row.iloc[0]['dec_lat_va']))
            lons.append(float(row.iloc[0]['dec_long_v']))
        else:
            lats.append(np.nan)
            lons.append(np.nan)

    # Load shapefile for panel (d)
    gdf = gpd.read_file(shp_path)
    
    # Create 2x2 subplot panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel (a): Gaussian perturbation mask
    im0 = axes[0,0].imshow(mask, cmap='YlOrBr')
    axes[0,0].set_title('(a) Gaussian Perturbation')
    fig.colorbar(im0, ax=axes[0,0])
    
    # Panel (b): Original Tmax field
    im1 = axes[0,1].imshow(original_field, cmap='YlOrBr')
    axes[0,1].set_title('(b) Original PPT')
    fig.colorbar(im1, ax=axes[0,1])
    
    # Panel (c): Perturbed Tmax field
    im2 = axes[1,0].imshow(perturbed_field, cmap='YlOrBr')
    axes[1,0].set_title('(c) Perturbed PPT')
    fig.colorbar(im2, ax=axes[1,0])
    
    # Panel (d): Sensitivity map with shapefile boundary and gauge points
    # Plot shapefile boundary
    gdf.boundary.plot(ax=axes[1,1], color='black', linewidth=0.3)
    # Plot gauge points colored by sensitivity
    sc = axes[1,1].scatter(lons, lats, c=sens, cmap='hot_r', s=100, edgecolors='k')
    axes[1,1].set_title('(d) Sensitivity Map')
    axes[1,1].set_xlabel('Longitude')
    axes[1,1].set_ylabel('Latitude')
    fig.colorbar(sc, ax=axes[1,1]).set_label('Sensitivity')
    axes[1,1].grid(True)
    # Optionally annotate gauge points
    # for i, (lon, lat) in enumerate(zip(lons, lats)):
    #     axes[1,1].text(lon, lat, f'{sens[i]:.2f}', fontsize=8, ha='right', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined panel plot: {save_path}")


def main():
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = CNN_LSTM().to(device)

    # checkpoint
    checkpoint_path = "/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/checkpoint/cnn_lstm_global_temporal_extrapolation_61_10y"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()   
    # data
    dataset = HDF5Dataset(config['h5_file'], ['ppt','tmin','tmax'], config['labels_path'], 2009, 2009)
    dataset_size = len(dataset) 
    # print(f"Dataset size: {dataset_size}")
    # num_train = int(0.8 * len(dataset))
    # num_val = int(0.1 * len(dataset))
    # num_test = len(dataset) - num_train - num_val
    # Randomly split the dataset
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_val, num_test])
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    # print(f"Test dataset size: {len(test_loader)}")
    
    # pick a single example from the loader
    # sample = next(iter(loader))
    sample = list(test_loader)[340]
    ppt = sample['ppt'].to(device)  # shape (1, T, H, W)
    tmin = sample['tmin'].to(device)
    tmax = sample['tmax'].to(device)

    # baseline: run inference on the full input (all sequences at once)
    orig_out = model(ppt, tmin, tmax).detach().cpu().numpy()

    # perturbation: apply perturbation to each frame, then combine them back into a sequence
    ppt_np = sample['ppt'].cpu().numpy()  # shape (1, T, H, W)
    tmin_np = sample['tmin'].cpu().numpy()
    tmax_np = sample['tmax'].cpu().numpy()
    _, T, H, W = ppt_np.shape
    ppt_pert_frames = []
    tmin_pert_frames = []
    tmax_pert_frames = []
    MASK = None
    # Use fixed perturbation parameters (you can modify per your needs)
    sigma = (150, 150)
    patch_size = 0
    amp = 1

    for t in range(T):
        # Select the t-th time slice from the single example
        ppt_slice = ppt_np[:, t:t+1, :, :]  # shape (1, 1, H, W)
        tmin_slice = tmin_np[:, t:t+1, :, :]
        tmax_slice = tmax_np[:, t:t+1, :, :]
        # Choose a random center for the perturbation (or set a fixed one)
        # y0 = np.random.randint(0, H)
        # x0 = np.random.randint(0, W)
        # center = (y0, x0)
        center = (1400, 500)
        
        # Apply perturbation on this slice
        ppt_p, mask = apply_gaussian_perturbation(ppt_slice, center, sigma=sigma, patch_size=patch_size, amplitude=amp)
        tmin_p, mask = apply_gaussian_perturbation(tmin_slice, center, sigma=sigma, patch_size=patch_size, amplitude=amp)
        tmax_p, mask = apply_gaussian_perturbation(tmax_slice, center, sigma=sigma, patch_size=patch_size, amplitude=amp)
        MASK = mask
        ppt_pert_frames.append(ppt_p)
        tmin_pert_frames.append(tmin_p)
        tmax_pert_frames.append(tmax_p)
    
    # Combine perturbed frames along the time axis; resulting shape: (1, T, H, W)
    pert_ppt = np.concatenate(ppt_pert_frames, axis=1)
    pert_tmin = np.concatenate(tmin_pert_frames, axis=1)
    pert_tmax = np.concatenate(tmax_pert_frames, axis=1)
    pert_ppt_tensor = torch.tensor(pert_ppt, dtype=torch.float32).to(device)
    pert_tmin_tensor = torch.tensor(pert_tmin, dtype=torch.float32).to(device)
    pert_tmax_tensor = torch.tensor(pert_tmax, dtype=torch.float32).to(device)
    # Run inference on the combined perturbed sequence
    pert_out = model(pert_ppt_tensor, pert_tmin_tensor, pert_tmax_tensor).detach().cpu().numpy()
    print(f"Perturbed output: {pert_out}")
    print(f"Original output: {orig_out}")
    # Compute sensitivity per gauge (or over the entire output)
    sens = calculate_sensitivity(orig_out, pert_out)
    print(f'Sensitivity shape: {sens.shape}')
    # For visualization, use the first time slice
    # orig_field = ppt_np[0, 0].copy()  # 2D field from the first sequence
    # mask2d = create_gaussian_patch_mask((H, W), center, patch_size, sigma, amp)
    # pert_field = orig_field + mask2d * orig_field.std()
    diff_map = np.full((H, W), sens.mean())

    save_fig = '/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/analysis/perturbation_technique/combined_panel.png'
    # For visualization, use the first time slice for original and perturbed fields

    
    
    
    site_numbers = [
        4099000, 4101500, 4097540, 4176000, 4097500, 4096515, 4176500, 4096405,
        4175600, 4102500, 4109000, 4106000, 4105500, 4167000, 4102700, 4166500,
        4104945, 4163400, 4117500, 4108600, 4112000, 4113000, 4108800, 4164100,
        4114000, 4160600, 4116000, 4144500, 4148500, 4146000, 4159900, 4118500,
        4147500, 4146063, 4115265, 4122100, 4151500, 4157005, 4121970, 4122200,
        4154000, 4152238, 4121500, 4122500, 4142000, 4124500, 4121300, 4125550,
        4126970, 4126740, 4127800, 4101800, 4105000, 4105700, 4112500, 4164300,
        4148140, 4115000, 4159492, 4121944, 4124200
    ]
    sensitivity_map_path = '/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/analysis/perturbation_technique/sensitivity_map.png'
    csv_path = '/home/talhamuh/water-research/CNN-LSMT/data/processed/streamflow_data/usgs_locations.csv'
    shp_path = "/home/talhamuh/water-research/CNN-LSMT/data/raw/Michigan/Final_Michigan_Map/Watershed_Boundary_Intersect_Michigan.shp"
    sensitivity_map_shp_path = '/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/analysis/perturbation_technique/sensitivity_map_shp.png'
    plot_combined_panel(MASK, ppt_np[0, 0], pert_ppt[0, 0], sens,
                        csv_path, site_numbers, shp_path, save_fig)
    
if __name__ == '__main__':
    main()
