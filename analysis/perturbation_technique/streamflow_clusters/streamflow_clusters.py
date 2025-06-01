import os
import fiona
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Ensure SHX restoration for shapefile
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# 1) Load watershed boundaries
shp_path = "/home/talhamuh/water-research/CNN-LSMT/data/raw/Michigan/Final_Michigan_Map/Watershed_Boundary_Intersect_Michigan.shp"
watersheds = []
with fiona.open(shp_path, 'r') as src:
    for feat in src:
        watersheds.append(shape(feat['geometry']))

# 2) Load and preprocess daily streamflow
df = pd.read_csv(
    "/home/talhamuh/water-research/CNN-LSMT/data/processed/streamflow_data/spatial_interpolation.csv",
    parse_dates=['date']
)
df['doy'] = df['date'].dt.dayofyear
df = df[df['doy'] != 366]  # drop Feb 29
stations = df.columns.difference(['date', 'doy'])
seasonal = df.groupby('doy')[stations].mean().T

# 3) Normalize seasonal curves
seasonal_norm = (
    seasonal
    .sub(seasonal.mean(axis=1), axis=0)
    .div(seasonal.std(axis=1), axis=0)
)
seasonal_norm.index = seasonal_norm.index.astype(int)

# 4) Load and normalize lat/lon
loc = pd.read_csv(
    '/home/talhamuh/water-research/CNN-LSMT/data/processed/streamflow_data/usgs_locations.csv'
).set_index('site_no')
lat = loc['dec_lat_va']
lon = loc['dec_long_v']
lat_norm = (lat - lat.mean()) / lat.std()
lon_norm = (lon - lon.mean()) / lon.std()

# 5) Build feature matrix X under three possible modes:
mode = 'Spatiotemporal'   # options: 'temporal_spatial', 'temporal', 'spatial'

if mode == 'Spatiotemporal':
    # 1) temporal + repeated lat/lon (half the length of your seasonal curve)
    features = []
    for sid in seasonal_norm.index:
        sf   = seasonal_norm.loc[sid].values                   # 365-day curve
        rep  = len(sf) // 2                                 # 365//2 == 182
        latv = np.repeat(lat_norm.loc[sid], rep)               # 182 copies of latitude
        lonv = np.repeat(lon_norm.loc[sid], rep)               # 182 copies of longitude
        features.append(np.concatenate([sf, latv, lonv]))      # → 365+182+182=729-dim
    X = np.vstack(features)

elif mode == 'temporal':
    # 2) purely temporal: one vector per station of length 365
    X = seasonal_norm.values

elif mode == 'spatial':
    # 3) purely spatial: one vector per station [mean, median, mode, std, lat, lon]
    # compute descriptive stats on the normalized seasonal curves
    mean_flow   = seasonal_norm.mean(axis=1)
    median_flow = seasonal_norm.median(axis=1)
    std_flow    = seasonal_norm.std(axis=1)
    mode_flow   = seasonal_norm.apply(lambda row: row.mode().iloc[0], axis=1)

    # choose how much to down-weight spatial coords (0.0 = ignore, 1.0 = unchanged)
    spatial_weight = 0.25

    lat_w = lat_norm.loc[mean_flow.index].values * spatial_weight
    lon_w = lon_norm.loc[mean_flow.index].values * spatial_weight

    # stack into feature matrix: [mean, median, mode, std, lat*0.5, lon*0.5]
    X = np.column_stack([
        mean_flow.values,
        median_flow.values,
        mode_flow.values,
        std_flow.values,
        lat_w,
        lon_w
    ])

else:
    raise ValueError(f"Unknown mode: {mode!r}. Must be 'temporal_spatial', 'temporal' or 'spatial'.")

# 6) Hierarchical clustering into 4 groups
Z = linkage(X, method='ward')
clusters = fcluster(Z, t=4, criterion='maxclust')

# -- EMBED and SAVE: Plot dendrogram for verification
plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    labels=[str(s) for s in seasonal_norm.index],
    color_threshold=Z[-3, 2],
    leaf_rotation=90
)
plt.axhline(y=Z[-4, 2], color='black', linestyle='--', label='4-cluster threshold')
plt.title(f'{mode} Ward Dendrogram (4 clusters)')
plt.xlabel('Station Number')
plt.ylabel('Linkage Distance')
plt.tight_layout()
dendrogram_path = f'{mode}_dendrogram_4_clusters.png'
plt.savefig(dendrogram_path, dpi=1000, bbox_inches='tight')
plt.show()
print(f"Dendrogram saved to {dendrogram_path}")

# 7) Prepare station DataFrame
df_pts = pd.DataFrame({
    'site_no': seasonal_norm.index.astype(int),
    'cluster': clusters,
    'lat': seasonal_norm.index.map(lat),
    'lon': seasonal_norm.index.map(lon),
})
df_pts['geometry'] = [Point(xy) for xy in zip(df_pts.lon, df_pts.lat)]

# select and rename
out_df = df_pts[['site_no','lon','lat','cluster']].rename(
    columns={'site_no':'station_id'}
)
csv_path = f'{mode}_station_clusters.csv'
out_df.to_csv(csv_path, index=False)
print(f"Station clusters written to {csv_path}")

# 8) Plot & save cluster map
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 10))
for poly in watersheds:
    if poly.geom_type == 'Polygon':
        xs, ys = poly.exterior.xy
        ax1.plot(xs, ys, color='black', lw=1)
    else:
        for part in poly.geoms:
            xs, ys = part.exterior.xy
            ax1.plot(xs, ys, color='black', lw=1)

colors = ['C0','C1','C2','C3']
for cid in range(1,5):
    sub = df_pts[df_pts.cluster == cid]
    ax1.scatter(
        sub.lon, sub.lat, c=colors[cid-1],
        label=f'Cluster {cid}', s=100, edgecolor='k', lw=0.3
    )
ax1.set_title(f'{mode} Streamflow Station Clusters (4 groups)', pad=12)
ax1.set_xlabel('Longitude'); ax1.set_ylabel('Latitude')
ax1.legend(loc='upper left', frameon=False)
plt.tight_layout()
cluster_map_path = f'{mode}_cluster_map_4.png'
plt.savefig(cluster_map_path, dpi=1000, bbox_inches='tight')
plt.close(fig1)
print(f"Cluster map saved to {cluster_map_path}")

# 9) Plot & save seasonal patterns in 2×2 grid with month labels
fig2, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()
months = pd.date_range('2001-01-01', periods=12, freq='MS')
xticks = months.dayofyear
xlabels = months.strftime('%b')
for idx, cid in enumerate(range(1,5)):
    ax = axes[idx]
    members = seasonal_norm.index[clusters == cid]
    # background grey curves
    for sid in members:
        ax.plot(
            np.arange(1, 366), seasonal_norm.loc[sid],
            color='gray', linewidth=0.5, alpha=0.4
        )
    # bold mean curve with original cluster colors
    mean_curve = seasonal_norm.loc[members].mean(axis=0)
    ax.plot(
        np.arange(1, 366), mean_curve,
        color=colors[cid-1], linewidth=2
    )
    ax.set_title(f'Cluster {cid}')
    ax.set_xlim(1, 365)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45)
    if idx % 2 == 0:
        ax.set_ylabel('Normalized discharge')
    if idx >= 2:
        ax.set_xlabel('Month')
plt.suptitle('Seasonal Patterns of 4 Streamflow Clusters', y=1.02, fontsize=14)
plt.tight_layout()
patterns_path = f'{mode}_cluster_patterns_4.png'
plt.savefig(patterns_path, dpi=1000, bbox_inches='tight')
plt.close(fig2)
print(f"Seasonal patterns saved to {patterns_path}")
