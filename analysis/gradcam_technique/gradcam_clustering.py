import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from kmeans_pytorch import kmeans
from pytorch_grad_cam import GradCAM
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml
import torch.nn.functional as F
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, MyMultiOutputTarget
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import skfuzzy as fuzz
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math




def run_kmean(features, n_clusters, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    return kmeans.fit_predict(features)

def run_gmm(features, n_clusters, device="cuda:1", seed=42):
    """
    Run Gaussian Mixture clustering on the given features.
    
    Parameters:
        features (array-like): The feature matrix.
        n_clusters (int): Number of mixture components.
        device (str): Device to use. Not used here since sklearn runs on cpu.
        seed (int): Random seed for reproducibility.
    
    Returns:
        labels: Cluster labels for each feature.
        means: Cluster centers.
    """
    gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
    gmm.fit(features)
    labels = gmm.predict(features)
    means = gmm.means_
    return labels, means

def run_fcm(features, n_clusters, m=1.2, error=1e-5, maxiter=100, seed=42):
    """
    Run fuzzy c-means clustering on the given features.
    
    Parameters:
        features (array-like): The feature matrix.
        n_clusters (int): Number of clusters.
        m (float): Fuzziness parameter.
        error (float): Error tolerance.
        maxiter (int): Maximum number of iterations.
        seed (int): Random seed for reproducibility.
    
    Returns:
        labels: Hard cluster labels for each feature.
        centers: Cluster centers.
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    # Transpose features: skfuzzy expects data of shape (n_features, n_samples)
    data = features.T  
    centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None, seed=seed
    )
    # u is the membership matrix with shape (n_clusters, n_samples)
    # Get hard assignments for each sample.
    labels = np.argmax(u, axis=0)
    return labels, centers

# ========================
num_locations = 61
# Later, load the array back from the file
location_features = np.load('gradcam_output.npy')


scaler = StandardScaler()
location_features = scaler.fit_transform(location_features)


# ========================
#  Set random seeds for reproducibility
# ========================
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define the cluster numbers to iterate over.
cluster_list = [2, 3, 4, 5, 6, 7, 8]

# List of site numbers corresponding to your locations 0 through 60
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

assert len(site_numbers) == 61, "Expected 61 site numbers."
clustering_technique = "fcm"
for n_clusters in cluster_list:
    print("=" * 40)
    print(f"Running analysis for {n_clusters} clusters")
    
    # Run k-means clustering on the location features.
    # location_labels = run_kmean(location_features, n_clusters, random_state=SEED)
    # location_labels, means = run_gmm(location_features, n_clusters=n_clusters, device="cuda:1", seed=42)
    
    location_labels, centers = run_fcm(
        location_features, 
        n_clusters=n_clusters, 
        m=1.2,          # adjust fuzziness parameter as desired
        error=1e-5,     # error tolerance
        maxiter=100,    # maximum number of iterations
        seed=42
    )
    
    print("Cluster assignments for each location:")
    for loc, label in enumerate(location_labels):
        print(f"Location {loc}: Cluster {label}")
    
    # Create mapping: Site Number -> Cluster
    site_cluster_map = {site: cluster for site, cluster in zip(site_numbers, location_labels)}
    
    print("Site Number -> Cluster Mapping:")
    for site, cluster in site_cluster_map.items():
        print(f"Site {site}: Cluster {cluster}")
    
    # Create a DataFrame for a neat table.
    df = pd.DataFrame({
        'Site Number': site_numbers,
        'Cluster': location_labels
    })
    print("\nMapping in tabular form:", df)
    
    usgs_path = "/home/talhamuh/water-research/CNN-LSMT/data/processed/streamflow_data/usgs_locations.csv"
    df_latlon = pd.read_csv(usgs_path)
    
    # Convert site_no to integer by removing leading zeros.
    df_latlon["site_no_int"] = df_latlon["site_no"].apply(lambda x: int(str(x).lstrip("0")))
    
    # Merge with your cluster DataFrame on "Site Number" and "site_no_int".
    df_merged = df.merge(
        df_latlon,
        left_on="Site Number",
        right_on="site_no_int",
        how="inner"
    )
    
    # Rename or drop columns as desired.
    df_merged.rename(columns={
        "Site Number": "site_number",
        "dec_lat_va": "latitude",
        "dec_long_v": "longitude",
        "Cluster": "cluster"
    }, inplace=True)
    df_merged.drop(columns=["site_no_int", "site_no"], inplace=True)
    
    final_df = df_merged[["site_number", "cluster", "longitude", "latitude"]].copy()
    
    import geopandas as gpd
    from matplotlib.colors import ListedColormap
    # Convert your DataFrame into a GeoDataFrame.
    geometry = gpd.points_from_xy(final_df['longitude'], final_df['latitude'])
    gdf_points = gpd.GeoDataFrame(final_df, geometry=geometry, crs="EPSG:4326")
    
    # Load the shapefile.
    shp_path = "/home/talhamuh/water-research/CNN-LSMT/data/raw/Michigan/Final_Michigan_Map/Watershed_Boundary_Intersect_Michigan.shp"
    gdf_map = gpd.read_file(shp_path)
    
    # Reproject the shapefile if necessary.
    if gdf_map.crs != "EPSG:4326":
        gdf_map = gdf_map.to_crs("EPSG:4326")
    
    # Get the bounding box: [min_long, min_lat, max_long, max_lat].
    min_long, min_lat, max_long, max_lat = gdf_map.total_bounds
    print("Min Longitude:", min_long)
    print("Min Latitude:", min_lat)
    print("Max Longitude:", max_long)
    print("Max Latitude:", max_lat)
    
    # Plot the base map and overlay the points.
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the shapefile as the base map.
    gdf_map.plot(ax=ax, color="none", edgecolor="gray")
    
    # Plot the USGS points colored by cluster.
    gdf_points.plot(ax=ax, column='cluster', cmap='ocean', markersize=80, edgecolor='k', legend=False)
    
    # Set plot and figure backgrounds to transparent.
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    
    # Save the plot.
    output_map_path = f"point_clusters/{n_clusters}_{clustering_technique}.png"
    plt.savefig(output_map_path, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Map saved to {output_map_path}")
    
    # Save the GeoDataFrame as a GeoJSON file.
    output_geojson_path = f"/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/analysis/point_clusters/{n_clusters}_{clustering_technique}_gdf_points.geojson"
    gdf_points.to_file(output_geojson_path, driver="GeoJSON")
    print(f"GeoDataFrame saved to {output_geojson_path}")
