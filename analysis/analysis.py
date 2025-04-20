import os
# 2. Set devices and seed
CUDA_VISIBLE_DEVICES = '1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
import sys
import yaml
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from shapely.geometry import shape
from affine import Affine
import rasterio.features
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
from kmeans_pytorch import kmeans
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import AgglomerativeClustering
import math
import skfuzzy as fuzz
import cupy as cp
# Ensure parent directory is in the path to import modules from ../
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loader import HDF5Dataset  # [src/cnn_lstm_project/data_loader.py]
# ---- CONFIGURABLE PARAMETERS ----
SEED = 42

N_CLUSTERS = [2, 3, 4, 5, 6, 7, 8]         # Number of clusters for individual variables (ppt, tmin, tmax)
# N_CLUSTERS = [2, 3]                     # Number of clusters for combined data
# ---------------------------------

def load_yaml_config(file_path):
    """Load YAML configuration from file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def set_seed(seed=SEED):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_kmeans(X, n_clusters, device, seed=SEED):
    """Reset seed and run k-means clustering."""
    set_seed(seed)
    return kmeans(X=X, num_clusters=n_clusters, distance='euclidean', device=device)

def run_agglomerative_clustering(X, n_clusters):
    """Perform Agglomerative Clustering."""
    agglom = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    return agglom.fit_predict(X)

def run_gmm_torch(X, n_clusters, max_iter=100, tol=1e-5, 
                            device='cuda', seed=42, batch_size=10000):
    """
    Perform Gaussian Mixture Model clustering using the EM algorithm with diagonal covariances,
    implemented in PyTorch for GPU acceleration with batched processing.
    
    Parameters:
      X            : Input data as a PyTorch tensor of shape (n_samples, n_features).
      n_components : Number of Gaussian components/clusters.
      max_iter     : Maximum number of iterations.
      tol          : Tolerance for convergence (relative improvement in log-likelihood).
      device       : Device to run on (e.g., 'cuda').
      seed         : Random seed for reproducibility.
      batch_size   : Number of samples to process per batch.
      
    Returns:
      labels       : Hard cluster assignments as a tensor of shape (n_samples,).
      means        : Learned means for each component (tensor of shape (n_components, n_features)).
    """
    # Set the random seed for reproducibility.
    torch.manual_seed(seed)
    X = X.to(device).float()
    n_samples, n_features = X.shape

    # Initialize mixture weights uniformly.
    pi = torch.ones(n_clusters, device=device) / n_clusters  # shape: (n_components,)
    
    # Initialize means by selecting random data points.
    indices = torch.randperm(n_samples)[:n_clusters]
    means = X[indices].clone()  # shape: (n_components, n_features)
    
    # Initialize variances using the variance of X along each feature.
    variance = X.var(dim=0, unbiased=False).unsqueeze(0).repeat(n_clusters, 1)
    variance = variance.clamp(min=1e-6)
    
    prev_ll = -float('inf')
    
    for iteration in range(max_iter):
        ll_sum = 0.0  # To accumulate log-likelihood over batches.
        # Accumulators for M-step:
        Nk = torch.zeros(n_clusters, device=device)
        sum_gamma_X = torch.zeros(n_clusters, n_features, device=device)
        sum_gamma_diff2 = torch.zeros(n_clusters, n_features, device=device)
        
        # --- E-step: process the data in batches ---
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = X[i:end]  # Shape: (B, n_features)
            B = X_batch.shape[0]
            
            # Expand dimensions:
            # X_batch_expand: (B, 1, n_features)
            # means_expand: (1, n_components, n_features)
            # var_expand: (1, n_components, n_features)
            X_batch_expand = X_batch.unsqueeze(1)
            means_expand = means.unsqueeze(0)
            var_expand = variance.unsqueeze(0)
            
            # Compute the log-likelihood of each point under each component.
            # log N(x; mu, sigma^2) = -0.5 * sum_d [ log(2*pi*sigma^2_d) + ((x_d - mu_d)^2 / sigma^2_d) ]
            log_prob = -0.5 * torch.sum(
                torch.log(2 * math.pi * var_expand) + ((X_batch_expand - means_expand) ** 2) / var_expand,
                dim=2
            )  # Shape: (B, n_components)
            log_prob = log_prob + torch.log(pi)  # Add mixing weights.
            
            # Use the log-sum-exp trick to compute responsibilities.
            max_log_prob, _ = torch.max(log_prob, dim=1, keepdim=True)  # (B, 1)
            log_sum_exp = max_log_prob + torch.log(torch.sum(torch.exp(log_prob - max_log_prob), dim=1, keepdim=True))  # (B, 1)
            
            # Compute log responsibilities and then responsibilities.
            log_gamma = log_prob - log_sum_exp  # Shape: (B, n_components)
            gamma = torch.exp(log_gamma)        # Shape: (B, n_components)
            
            # Accumulate log-likelihood (over this batch).
            ll_sum += torch.sum(log_sum_exp)
            
            # Update accumulators for the M-step.
            Nk += torch.sum(gamma, dim=0)                               # (n_components,)
            sum_gamma_X += gamma.t() @ X_batch                          # (n_components, n_features)
            # For variance: accumulate gamma * (X - means)^2
            diff = X_batch.unsqueeze(1) - means.unsqueeze(0)            # Shape: (B, n_components, n_features)
            sum_gamma_diff2 += torch.sum(gamma.unsqueeze(2) * diff**2, dim=0)  # (n_components, n_features)
            
        ll = ll_sum  # Total log-likelihood for current iteration.
        
        # --- M-step: update parameters using the accumulated values ---
        pi = Nk / n_samples
        means_new = sum_gamma_X / Nk.unsqueeze(1)
        variance_new = sum_gamma_diff2 / Nk.unsqueeze(1)
        variance_new = variance_new.clamp(min=1e-6)
        
        # Check for convergence (relative improvement).
        if abs(ll - prev_ll) < tol * abs(ll):
            print(f"Converged at iteration {iteration+1}, log-likelihood: {ll.item()}")
            means = means_new
            variance = variance_new
            break
        
        prev_ll = ll
        means = means_new
        variance = variance_new
        print(f"Iteration {iteration+1}, log-likelihood: {ll.item()}")
    
    # --- Final: compute hard assignments in batches ---
    all_labels = torch.empty(n_samples, dtype=torch.long, device=device)
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        X_batch = X[i:end]
        # Expand dimensions for current parameters.
        X_batch_expand = X_batch.unsqueeze(1)         # (B, 1, n_features)
        means_expand = means.unsqueeze(0)             # (1, n_components, n_features)
        var_expand = variance.unsqueeze(0)            # (1, n_components, n_features)
        
        # Compute log probabilities.
        log_prob = -0.5 * torch.sum(
            torch.log(2 * math.pi * var_expand) + ((X_batch_expand - means_expand)**2) / var_expand,
            dim=2
        )
        log_prob = log_prob + torch.log(pi)
        # Hard assignment: maximum log probability for each sample.
        _, labels_batch = torch.max(log_prob, dim=1)
        all_labels[i:end] = labels_batch
        
    return all_labels, means

def fuzzy_c_means(X, n_clusters, m=1.2, max_iter=100, tol=1e-5, 
                          device="cuda", seed=42, batch_size=10000):
    """
    Perform Fuzzy C-Means clustering using GPU and batched computations 
    to mitigate out-of-memory issues.

    Parameters:
      X           : Input data, a tensor of shape (n_samples, n_features).
                    (Large data should be on CPU; it will be moved to the GPU
                    in a batched fashion.)
      n_clusters  : Number of clusters.
      m           : Fuzziness parameter (default 1.5; try values close to 1.5-2.0).
      max_iter    : Maximum iterations.
      tol         : Convergence tolerance.
      device      : GPU device to use (e.g., "cuda:1").
      seed        : Random seed.
      batch_size  : Size of batch for processing; adjust based on GPU memory.

    Returns:
      labels      : Hard assignments for each point (tensor of shape [n_samples]).
      centers     : Final cluster centers (tensor of shape [n_clusters, n_features]).
    """
    set_seed(seed)

    # Move X to the desired device in float32 if possible.
    X = X.to(device).float()
    n_samples, n_features = X.shape

    # Initialize membership matrix U of shape (n_samples, n_clusters).
    U = torch.rand(n_samples, n_clusters, device=device)
    U = U / U.sum(dim=1, keepdim=True)

    # Helper to compute centers in a batched way.
    def compute_centers(X, U, m):
        # We'll accumulate numerator and denominator in CPU or GPU.
        num = torch.zeros(n_clusters, n_features, device=device)
        den = torch.zeros(n_clusters, device=device)
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = X[i:end]            # (B, n_features)
            U_batch = U[i:end]            # (B, n_clusters)
            um = U_batch ** m             # (B, n_clusters)
            num += um.transpose(0, 1) @ X_batch  # (n_clusters, n_features)
            den += um.sum(dim=0)                 # (n_clusters,)
        centers = num / den.unsqueeze(1)
        return centers

    # Helper to update membership matrix U in batches.
    def update_memberships(X, centers, exponent):
        new_U = torch.empty(n_samples, n_clusters, device=device)
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = X[i:end]  # (B, n_features)
            # Compute distances between X_batch and centers: (B, n_clusters)
            # Using torch.cdist in a batched way avoids huge allocations.
            d_batch = torch.cdist(X_batch, centers, p=2) + 1e-8  # shape (B, n_clusters)
            # For each sample in batch, update membership:
            # ratio: (B, n_clusters, n_clusters)
            ratio = (d_batch.unsqueeze(2) / d_batch.unsqueeze(1)).pow(exponent)
            # Sum over the last dimension: (B, n_clusters)
            new_U_batch = 1.0 / ratio.sum(dim=2)
            new_U[i:end] = new_U_batch
        return new_U

    exponent = 2.0 / (m - 1)
    
    # Main iterative loop
    for iteration in range(max_iter):
        U_old = U.clone()
        # Compute new centers in a batched way
        centers = compute_centers(X, U, m)
        # Update membership matrix U in batches
        U = update_memberships(X, centers, exponent)

        # Compute convergence criteria (norm over full U)
        norm_diff = torch.norm(U - U_old)
        print(f"Iteration {iteration+1}: Norm diff = {norm_diff.item():.6f}")
        if norm_diff < tol:
            print("Convergence reached.")
            break

    # Return hard assignments and cluster centers.
    labels = U.argmax(dim=1)
    return labels, centers  

def plot_cluster_results(cluster_arrs, titles, cmap, figsize=(18, 6), save_path="cluster_results.png"):
    """Plot list of 2D cluster arrays in subplots and save the figure locally."""
    n = len(cluster_arrs)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, clust, title in zip(axes, cluster_arrs, titles):
        img = ax.imshow(clust, cmap='ocean')
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(img, ax=ax, orientation='vertical', label="Cluster ID")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 1. Load configuration
    config = load_yaml_config('../config/config.yaml')

   
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 3. Load the dataset using HDF5Dataset
    variables_to_load = ['ppt', 'tmin', 'tmax']
    dataset = HDF5Dataset(
        config['h5_file'], variables_to_load, config['labels_path'],
        2000, 2009, sequence_length=1
    )
    dataset_size = len(dataset)
    num_train = int(0.8 * dataset_size)
    num_val = int(0.1 * dataset_size)
    test_dataset = Subset(dataset, range(num_train + num_val, dataset_size))
    

    test_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=32)

    # 4. Load data and remove the single channel dimension
    all_ppt, all_tmin, all_tmax = [], [], []
    for data_dict in test_loader:
        all_ppt.append(data_dict['ppt'].squeeze(1))
        all_tmin.append(data_dict['tmin'].squeeze(1))
        all_tmax.append(data_dict['tmax'].squeeze(1))
    ppt_all  = torch.cat(all_ppt,  dim=0)
    tmin_all = torch.cat(all_tmin, dim=0)
    tmax_all = torch.cat(all_tmax, dim=0)
    
    ppt_all = ppt_all.reshape(10, 365, 1849, 1458)
    tmin_all = tmin_all.reshape(10, 365, 1849, 1458)
    tmax_all = tmax_all.reshape(10, 365, 1849, 1458)
    
    
    print("ppt_all shape:", ppt_all.shape)
    print("tmin_all shape:", tmin_all.shape)
    print("tmax_all shape:", tmax_all.shape)

    
    ppt_daily_mean = ppt_all.mean(axis=0)
    tmin_daily_mean = tmin_all.mean(axis=0)
    tmax_daily_mean = tmax_all.mean(axis=0)
    
    print(ppt_daily_mean.shape)
    print(tmin_daily_mean.shape)
    print(tmax_daily_mean.shape)
    
    # 5. Convert tensor data to NumPy arrays and reshape (H, W, T)
    ppt_np = ppt_daily_mean.cpu().numpy()
    tmin_np = tmin_daily_mean.cpu().numpy()
    tmax_np = tmax_daily_mean.cpu().numpy()
    ppt_transposed  = np.transpose(ppt_np, (1, 2, 0))
    tmin_transposed = np.transpose(tmin_np, (1, 2, 0))
    tmax_transposed = np.transpose(tmax_np, (1, 2, 0))
    H, W, T = ppt_transposed.shape
    ppt_flat  = ppt_transposed.reshape(H * W, T)
    tmin_flat = tmin_transposed.reshape(H * W, T)
    tmax_flat = tmax_transposed.reshape(H * W, T)
    combined_data_flat = np.concatenate((ppt_flat, tmin_flat, tmax_flat), axis=1)
    print("data_flat shape (ppt):", ppt_flat.shape)
    print("data_flat shape (tmin):", tmin_flat.shape)
    print("data_flat shape (tmax):", tmax_flat.shape)
    print("combined_data_flat shape:", combined_data_flat.shape)

    # 6. Standardize the flattened data
    scaler = StandardScaler()
    ppt_flat_scaled  = scaler.fit_transform(ppt_flat)
    tmin_flat_scaled = scaler.fit_transform(tmin_flat)
    tmax_flat_scaled = scaler.fit_transform(tmax_flat)
    combined_data_flat_scaled = scaler.fit_transform(combined_data_flat)

    # 7. Perform k-means clustering on each variable using device_cluster
    device_cluster = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ppt_torch  = torch.from_numpy(ppt_flat_scaled).float().to(device_cluster)
    tmin_torch = torch.from_numpy(tmin_flat_scaled).float().to(device_cluster)
    tmax_torch = torch.from_numpy(tmax_flat_scaled).float().to(device_cluster)
    combined_data_torch = torch.from_numpy(combined_data_flat_scaled).float().to(device_cluster)
    

    # Your provided geographic bounds and CRS
    max_lon, min_lon = -82.4198362733253, -86.89262785928825
    max_lat, min_lat = 45.850304369950784, 41.6904825223513
    CRS = "EPSG:4326"

    technique_region = ["kmeans", "gmm", "fcm"]
    technique_point_clustering = ["kmeans", "gmm", "fcm"]
    # First, compute all clusters once for each technique and N_CLUSTERS value.
    precomputed_clusters = {}
    for technique in technique_region:
        precomputed_clusters[technique] = {}
        for n_clusters in N_CLUSTERS:
            if technique == "kmeans":
                ppt_cluster_ids, _ = run_kmeans(ppt_torch, n_clusters + 1, device_cluster, 42)
                tmin_cluster_ids, _ = run_kmeans(tmin_torch, n_clusters + 1, device_cluster, 42)
                tmax_cluster_ids, _ = run_kmeans(tmax_torch, n_clusters + 1, device_cluster, 42)
                combined_cluster_ids, _ = run_kmeans(combined_data_torch, n_clusters + 1, device, 42)
            elif technique == "fcm":
                ppt_cluster_ids, _ = fuzzy_c_means(ppt_torch, n_clusters=n_clusters+1, device="cuda:1", seed=42)
                tmin_cluster_ids, _ = fuzzy_c_means(tmin_torch, n_clusters=n_clusters+1, device="cuda:1", seed=42)
                tmax_cluster_ids, _ = fuzzy_c_means(tmax_torch, n_clusters=n_clusters+1, device="cuda:1", seed=42)
                combined_cluster_ids, _ = fuzzy_c_means(combined_data_torch, n_clusters=n_clusters+1, device="cuda:1", seed=42)
            elif technique == "gmm":
                ppt_cluster_ids, _ = run_gmm_torch(ppt_torch, n_clusters=n_clusters+1, device="cuda:1", seed=42)
                tmin_cluster_ids, _ = run_gmm_torch(tmin_torch, n_clusters=n_clusters+1, device="cuda:1", seed=42)
                tmax_cluster_ids, _ = run_gmm_torch(tmax_torch, n_clusters=n_clusters+1, device="cuda:1", seed=42)
                combined_cluster_ids, _ = run_gmm_torch(combined_data_torch, n_clusters=n_clusters+1, device="cuda:1", seed=42)
            # Correct usage of np.unique to get both unique values and their counts

            precomputed_clusters[technique][n_clusters] = {
                "ppt": ppt_cluster_ids,
                "tmin": tmin_cluster_ids,
                "tmax": tmax_cluster_ids,
                "combined": combined_cluster_ids
            }

    # Next, process the precomputed clustering results for each technique and point clustering method.
    for technique, technique_clusters in precomputed_clusters.items():
        for pct in technique_point_clustering:
            # Create a dict to accumulate pivot DataFrames for each n_clusters value.
            pivot_dict = {n_clusters: [] for n_clusters in N_CLUSTERS}

            # Process each n_clusters value.
            for n_clusters in N_CLUSTERS:
                clusters = technique_clusters[n_clusters]

                ppt_labels_np = clusters["ppt"].cpu().numpy()
                tmin_labels_np = clusters["tmin"].cpu().numpy()
                tmax_labels_np = clusters["tmax"].cpu().numpy()
                combined_labels_np = clusters["combined"].cpu().numpy()

                # Reshape arrays back to (H, W).
                ppt_cluster_np = ppt_labels_np.reshape(H, W)
                tmin_cluster_np = tmin_labels_np.reshape(H, W)
                tmax_cluster_np = tmax_labels_np.reshape(H, W)
                combined_data_clusters_np = combined_labels_np.reshape(H, W)

                clusters_dict = {
                    "ppt": ppt_cluster_np,
                    "tmin": tmin_cluster_np,
                    "tmax": tmax_cluster_np,
                    "combined": combined_data_clusters_np
                }

                # Process each variable's clustering result.
                for key, cluster_np in clusters_dict.items():
                    H_cluster, W_cluster = cluster_np.shape

                    # Compute pixel resolutions.
                    xres = (max_lon - min_lon) / W_cluster
                    yres = (max_lat - min_lat) / H_cluster
                    transform = Affine.translation(min_lon, max_lat) * Affine.scale(xres, -yres)

                    # Vectorize the raster clusters to polygons.
                    shapes_and_values = list(rasterio.features.shapes(cluster_np.astype(np.int32), transform=transform))
                    geoms = []
                    cluster_ids_list = []
                    for geom, value in shapes_and_values:
                        geoms.append(shape(geom))
                        cluster_ids_list.append(value)
                    gdf = gpd.GeoDataFrame({'cluster_id': cluster_ids_list, 'geometry': geoms}, crs=CRS)

                    # Read corresponding point clusters from geojson.
                    geojson_path = f'/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/analysis/point_clusters/{n_clusters}_{pct}_gdf_points.geojson'
                    geojson_gdf = gpd.read_file(geojson_path)
                    shp_path = "/home/talhamuh/water-research/CNN-LSMT/data/raw/Michigan/Final_Michigan_Map/Watershed_Boundary_Intersect_Michigan.shp"
                    gdf_map = gpd.read_file(shp_path)

                    # Ensure same CRS.
                    gdf = gdf.to_crs("EPSG:4326")
                    geojson_gdf = geojson_gdf.to_crs("EPSG:4326")
                    gdf_map = gdf_map.to_crs("EPSG:4326")

                    # Spatial join and error computation.
                    total_counts = geojson_gdf.groupby("cluster").size().reset_index(name="total_count")
                    joined = gpd.sjoin(geojson_gdf, gdf, how="left", predicate="within")
                    region_counts = joined.groupby(["cluster_id", "cluster"]).size().reset_index(name="count_in_region")
                    region_ids = gdf["cluster_id"].unique()
                    point_clusters = geojson_gdf["cluster"].unique()
                    cartesian = pd.DataFrame(list(itertools.product(region_ids, point_clusters)),
                                             columns=["cluster_id", "cluster"])
                    merged = pd.merge(cartesian, region_counts, on=["cluster_id", "cluster"], how="left")
                    merged["count_in_region"] = merged["count_in_region"].fillna(0)
                    merged = pd.merge(merged, total_counts, on="cluster", how="left")
                    merged["error"] = (merged["total_count"] - merged["count_in_region"]) / merged["total_count"]
                    # Filter out regions with zero count.
                    merged = merged.groupby('cluster_id').filter(lambda grp: grp['count_in_region'].sum() > 0)
                    # Reset region IDs sequentially
                    merged['cluster_id'] = merged.groupby('cluster_id').ngroup()
                    # Create a pivot table.
                    pivot_df = merged.pivot(index='cluster', columns='cluster_id', values='error')
                    pivot_df = pivot_df.reset_index().rename(columns={'index': 'cluster'})
                    pivot_df.insert(0, 'variable', key)
                    pivot_df.insert(1, 'n_clusters', n_clusters)

                    # Append this pivot DataFrame to our list for current n_clusters.
                    pivot_dict[n_clusters].append(pivot_df)

            # Write to Excel: Each sheet corresponds to a value from N_CLUSTERS,
            # and each sheet contains the stacked pivot tables for all variables.
            excel_file = f"r_{technique}_p_{pct}_results.xlsx"
            with pd.ExcelWriter(excel_file) as writer:
                for n in sorted(pivot_dict.keys()):
                    if pivot_dict[n]:
                        final_pivot_df = pd.concat(pivot_dict[n], ignore_index=True)
                    else:
                        final_pivot_df = pd.DataFrame()
                    sheet_name = f"{n}_clusters"
                    final_pivot_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Saved combined pivot tables to {excel_file}")

if __name__ == '__main__':
    main()