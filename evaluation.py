import os
import yaml
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from data_loader import HDF5Dataset
from main import setup_logging, load_checkpoint, calculate_nse  # Reuse functions from main.py
from model import CNN_LSTM  # Adjust import if your model lives elsewhere

# Global constant for site numbers (order must match model outputs)
SITE_NUMBERS = [
    4099000, 4101500, 4097540, 4176000, 4097500, 4096515, 4176500, 4096405,
    4175600, 4102500, 4109000, 4106000, 4105500, 4167000, 4102700, 4166500,
    4104945, 4163400, 4117500, 4108600, 4112000, 4113000, 4108800, 4164100,
    4114000, 4160600, 4116000, 4144500, 4148500, 4146000, 4159900, 4118500,
    4147500, 4146063, 4115265, 4122100, 4151500, 4157005, 4121970, 4122200,
    4154000, 4152238, 4121500, 4122500, 4142000, 4124500, 4121300, 4125550,
    4126970, 4126740, 4127800, 4101800, 4105000, 4105700, 4112500, 4164300,
    4148140, 4115000, 4159492, 4121944, 4124200
]

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# keep your existing imports for colormaps, logger, calculate_nse, etc.
from data_loader import HDF5Dataset
from main import calculate_nse
from model import CNN_LSTM

# ... your SITE_NUMBERS, etc.

def inference(model, dataloader, device, save_dir, logger):
    """
    Runs the model in eval mode over the dataloader, computes MSE/NSE/R2/KGE/PBIAS per station,
    saves time‐series + error plots, and writes out an Excel sheet of all metrics.
    """
    # 1) Styling and colormaps (unchanged) …
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor':   'white',
        'axes.edgecolor':   'black',
        'grid.color':       'lightgray',
        'grid.linestyle':   '--',
        'grid.alpha':       0.3,
        'lines.linewidth':  1.0,
    })
    cmap_tab   = plt.get_cmap('tab10')
    actual_col = cmap_tab(0)
    pred_col   = cmap_tab(1)
    cmap_div   = plt.get_cmap('RdBu_r')

    model.eval()
    all_outputs = []
    all_labels  = []

    # 2) Forward‐pass (unchanged) …
    with torch.no_grad():
        for batch in dataloader:
            ppt   = batch['ppt'].to(device)
            tmin  = batch['tmin'].to(device)
            tmax  = batch['tmax'].to(device)
            label = batch['label'].to(device)

            out = model(ppt, tmin, tmax)
            all_outputs.append(out)
            all_labels.append(label)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels  = torch.cat(all_labels,  dim=0)

    # 3) Prepare output dirs (unchanged) …
    os.makedirs(save_dir, exist_ok=True)
    hist_dir = os.path.join(save_dir, 'error_histograms')
    os.makedirs(hist_dir, exist_ok=True)

    # 4) Build a DateTimeIndex (unchanged) …
    dates = pd.date_range(start='2009-01-01',
                          periods=all_outputs.shape[0],
                          freq='D')

    # 5) Initialize metric lists (added kge_vals, pbias_vals)
    mse_vals, nse_vals, r2_vals = [], [], []
    kge_vals, pbias_vals = [], []
    site_ids = []

    for i in range(all_outputs.shape[1]):
        obs  = all_labels[:, i].cpu().numpy()
        pred = all_outputs[:, i].cpu().numpy()

        # MSE, NSE, R²
        mse = ((pred - obs)**2).mean()
        nse = calculate_nse(torch.from_numpy(obs), torch.from_numpy(pred)).item()
        r2  = __import__('sklearn.metrics').metrics.r2_score(obs, pred)

        # --- NEW: KGE calculation ---
        # r: Pearson’s correlation
        r = np.corrcoef(pred, obs)[0, 1]
        # alpha: ratio of standard deviations
        alpha = np.std(pred) / np.std(obs)
        # beta: ratio of means
        beta = np.mean(pred) / np.mean(obs)
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        # --- NEW: PBIAS calculation ---
        # percent bias = 100 * sum(pred - obs) / sum(obs)
        pbias = 100.0 * (np.sum(pred - obs) / np.sum(obs))

        # Append to lists
        mse_vals.append(mse)
        nse_vals.append(nse)
        r2_vals.append(r2)
        kge_vals.append(kge)
        pbias_vals.append(pbias)

        site_id = SITE_NUMBERS[i] if i < len(SITE_NUMBERS) else i+1
        site_ids.append(site_id)

        logger.info(
            f"Site {site_id}: R²={r2:.3f}, NSE={nse:.3f}, "
            f"KGE={kge:.3f}, PBIAS={pbias:.2f}%, MSE={mse:.3f}"
        )

        # —— Time‐series plot —— (unchanged) …
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, obs,  label='Observed',   color=actual_col)
        ax.plot(dates, pred, label='Predicted', linestyle='--', color=pred_col)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        fig.autofmt_xdate()
        ax.set_title(f"Streamflow at Site {site_id}", fontsize=14)
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Discharge", fontsize=12)
        ax.legend(frameon=True)
        ax.grid(True)
        ax.text(0.02, 0.82, f"NSE = {nse:.2f}\nKGE = {kge:.2f}", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"timeseries_site_{site_id}.png"), dpi=1000)
        plt.close(fig)

        # —— Percent‐error histogram —— (unchanged) …
        percent_error = 100 * (pred - obs) / (obs + 1e-8)
        max_err = abs(percent_error).max()
        norm    = plt.Normalize(-max_err, max_err)
        colors  = cmap_div(norm(percent_error))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(dates, percent_error, color=colors, width=2)
        ax.axhline(0, color='black', linewidth=1)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        fig.autofmt_xdate()
        ax.set_title(f"Percent Error at Site {site_id}", fontsize=14)
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Error (%)", fontsize=12)
        ax.grid(True)
        sm = plt.cm.ScalarMappable(cmap=cmap_div, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Percent Error (%)")
        fig.tight_layout()
        fig.savefig(os.path.join(hist_dir, f"error_hist_site_{site_id}.png"), dpi=1000)
        plt.close(fig)

    # 6) Save all metrics to Excel
    df_metrics = pd.DataFrame({
        'site_id': site_ids,
        'MSE':     mse_vals,
        'NSE':     nse_vals,
        'R2':      r2_vals,
        'KGE':     kge_vals,
        'PBIAS (%)': pbias_vals
    })
    excel_path = os.path.join(save_dir, 'evaluation_metrics.xlsx')
    df_metrics.to_excel(excel_path, index=False)
    logger.info(f"Written evaluation metrics to {excel_path}")

    return {
        'mse':   mse_vals,
        'nse':   nse_vals,
        'r2':    r2_vals,
        'kge':   kge_vals,
        'pbias': pbias_vals
    }

def plot_cluster_nse_boxplots(nse_values, csv_path, save_dir, logger):
    """
    Reads the station cluster csv file, maps each site to its cluster, and then creates
    box plots of NSE values grouped by cluster.
    """
    # Read CSV file containing station clusters
    try:
        clusters_df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return

    # Create a mapping from station_id to cluster (ensure station ids are integers)
    station_to_cluster = clusters_df.set_index('station_id')['cluster'].to_dict()

    # Group NSE values per cluster using the global SITE_NUMBERS list for mapping order
    cluster_nse = {}  # { cluster: [nse, ...] }
    for i, station in enumerate(SITE_NUMBERS):
        if station in station_to_cluster:
            cluster = station_to_cluster[station]
            cluster_nse.setdefault(cluster, []).append(nse_values[i])
        else:
            logger.warning(f"Station {station} not found in cluster mapping.")

    # Prepare data for boxplot: sort clusters for consistent ordering
    clusters_sorted = sorted(cluster_nse.keys())
    data_to_plot = [cluster_nse[cl] for cl in clusters_sorted]

    # Create box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=[f'Cluster {cl}' for cl in clusters_sorted], patch_artist=True)
    plt.ylabel('NSE')
    plt.title('NSE Values by Cluster')
    plt.grid(axis='y', alpha=0.75)

    plot_path = os.path.join(save_dir, 'nse_boxplots_by_cluster_2009.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"NSE box plot by cluster saved to {plot_path}")


def plot_nse_by_flow_category(nse_values, excel_path, save_dir, logger):
    """
    Reads the Excel file with station flow categories (‘Station’, ‘Category’),
    maps each SITE_NUMBERS entry to its category, and creates
    a boxplot of NSE values grouped by Low/Middle/High flow.
    """
    # 1. Load the Excel with your pre-computed categories
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        return

    # 2. Validate columns
    if 'Station' not in df.columns or 'Category' not in df.columns:
        logger.error("Excel must contain 'Station' and 'Category' columns.")
        return

    # 3. Build station → category map
    station_to_cat = df.set_index('Station')['Category'].to_dict()

    # 4. Group NSE values by category
    category_nse = {}
    for idx, station in enumerate(SITE_NUMBERS):
        cat = station_to_cat.get(station)
        if cat is None:
            logger.warning(f"Station {station} not in flow‐categories file.")
            continue
        category_nse.setdefault(cat, []).append(nse_values[idx])

    # 5. Prepare data in a fixed order
    categories = ['Low', 'Middle', 'High']
    data_to_plot = [category_nse.get(cat, []) for cat in categories]

    # 6. Plot and save
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot,
                tick_labels=categories,   # use tick_labels instead of deprecated labels=
                patch_artist=True)
    plt.xlabel('Flow Category')
    plt.ylabel('NSE')
    plt.title('NSE Distribution by Flow Category')
    plt.grid(axis='y', alpha=0.75)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'nse_boxplots_by_flow_category.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    logger.info(f"NSE‐by‐flow‐category boxplot saved to {out_path}")

        

def main():
    # Load configuration from YAML file
    config = load_yaml_config('config/config.yaml')
    
    # Set up logging via the shared function
    setup_logging('config/logging_config.yaml')
    logger = logging.getLogger('my_application')
    
    # Set device; uses first GPU specified in config if available
    device = torch.device(f"cuda:{config['gpu'][0]}" if torch.cuda.is_available() else "cpu")
    
    # Prepare the dataset for evaluation
    variables_to_load = ['ppt', 'tmin', 'tmax']
    dataset = HDF5Dataset(config['h5_file'], variables_to_load, config['labels_path'], 2009, 2009)
    logger.info(f"Evaluation dataset size: {len(dataset)}")
    
    eval_loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=32, shuffle=False)
    
    model = CNN_LSTM().to(device)
    model = nn.DataParallel(model, device_ids=config['gpu'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    model, optimizer, _, _ = load_checkpoint(config['checkpoint_path'], model, optimizer, device)
    
    results_dir = os.path.join('results', 'evaluation')
    os.makedirs(results_dir, exist_ok=True)
    
    # Get metrics from inference (including NSE values)
    metrics = inference(model, eval_loader, device, results_dir, logger)
    
    # Path to the CSV file containing station clusters
    csv_path = config['usgs_clusters']
    
    # Plot box plots for NSE values grouped by cluster
    plot_cluster_nse_boxplots(metrics['nse'], csv_path, results_dir, logger)
    

    plot_nse_by_flow_category(
        nse_values=metrics['nse'],
        excel_path='/home/talhamuh/water-research/CNN-LSMT/data/processed/streamflow_data/location_flow_categories.xlsx',
        save_dir=results_dir,
        logger=logger
    )
if __name__ == '__main__':
    main()