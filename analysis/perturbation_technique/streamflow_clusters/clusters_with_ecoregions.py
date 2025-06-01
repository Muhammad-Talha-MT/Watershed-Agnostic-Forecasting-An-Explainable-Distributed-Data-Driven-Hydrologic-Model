#!/usr/bin/env python3
import fiona
from shapely.geometry import shape, Point
from shapely.ops import unary_union, transform as shp_transform
from pyproj import CRS, Transformer
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Patch
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency
import os
# --------------------------------------------------------------------------
# 0. Paths — update these as needed
# --------------------------------------------------------------------------

mode = 'temporal'  # options: 'spatiotemporal', 'temporal', 'spatial'
roi_path     = "/home/talhamuh/water-research/CNN-LSMT/data/raw/Michigan/Final_Michigan_Map/Watershed_Boundary_Intersect_Michigan.shp"
base_path = "/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/analysis/perturbation_technique/weather_clusters"
region_path  = [
                    "/home/talhamuh/water-research/CNN-LSMT/data/raw/us_eco_l3_state_boundaries/us_eco_l3_state_boundaries.shp"
            ]

results_file = "Clusters_x_regions/cluster_stats.xlsx"
stations_csv = f"/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/analysis/perturbation_technique/streamflow_clusters/{mode}_station_clusters.csv"
# Output files
table_out    = f"{mode}_cluster_by_ecoregion.csv"
fig_out      = f"{mode}_Clusters_vs_Ecoregions.png"

# --------------------------------------------------------------------------
# 1. Load & union ROI (ensuring EPSG:4326)
# --------------------------------------------------------------------------
roi_parts = []
with fiona.open(roi_path) as src:
    src_crs = CRS.from_wkt(src.crs_wkt)
    if src_crs.to_epsg() != 4326:
        to_wgs = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True).transform
    else:
        to_wgs = None

    for feat in src:
        g = shape(feat["geometry"])
        if to_wgs:
            g = shp_transform(to_wgs, g)
        roi_parts.append(g)
roi_union = unary_union(roi_parts)

# --------------------------------------------------------------------------
# 2. Load, transform & clip Michigan ecoregions from multiple shapefiles
# --------------------------------------------------------------------------
eco_geoms, eco_names = [], []
with fiona.open(region_path[0]) as src:
    eco_crs = CRS.from_wkt(src.crs_wkt)
    to_wgs  = Transformer.from_crs(eco_crs, "EPSG:4326", always_xy=True).transform
    for feat in src:
        if feat["properties"].get("STATE_NAME") == "Michigan":
            g = shp_transform(to_wgs, shape(feat["geometry"]))
            clip = g.intersection(roi_union)
            if not clip.is_empty:
                eco_geoms.append(clip)
                eco_names.append(feat["properties"]["US_L3NAME"])

# --------------------------------------------------------------------------
# 3. Load stations & assign ecoregions
# --------------------------------------------------------------------------
rows = []
df = pd.read_csv(stations_csv)
for _, row in df.iterrows():
    lon, lat, cl = row["lon"], row["lat"], int(row["cluster"])
    pt = Point(lon, lat)
    if not roi_union.contains(pt):
        continue
    # find containing ecoregion
    name = next((nm for g, nm in zip(eco_geoms, eco_names) if g.contains(pt)), None)
    rows.append({"cluster": cl, "ecoregion": name, "lon": lon, "lat": lat})
df_st = pd.DataFrame(rows)
if df_st.empty:
    raise RuntimeError("No stations fell inside the ROI!")

# --------------------------------------------------------------------------
# 4. Contingency table + statistics
# --------------------------------------------------------------------------
table = pd.crosstab(df_st["cluster"], df_st["ecoregion"])
table.to_csv(table_out)

chi2, p, dof, _ = chi2_contingency(table.fillna(0))
N = table.values.sum()
cramer_v = (chi2 / (N * (min(table.shape)-1))) ** 0.5

print("\n=== Cluster × Ecoregion Contingency Table ===\n")
print(table)
print(f"\n(table saved to: {table_out})")
print(f"\nChi² = {chi2:.2f} (df={dof}), p = {p:.3g}")
print(f"Cramer's V = {cramer_v:.3f}\n")

# --------------------------------------------------------------------------
# 5. Plot filled ecoregions + stations + legends
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 10))

# a) Pastel fills for Michigan ecoregions
unique_names = list(dict.fromkeys(eco_names))
cmap = plt.get_cmap("tab20", len(unique_names))
fill_colors = {nm: cmap(i) for i, nm in enumerate(unique_names)}

for geom, nm in zip(eco_geoms, eco_names):
    parts = [geom] if geom.geom_type == "Polygon" else geom.geoms
    for part in parts:
        poly = MplPolygon(
            list(part.exterior.coords),
            facecolor=fill_colors[nm],
            edgecolor="grey",
            linewidth=0.5,
            alpha=0.3
        )
        ax.add_patch(poly)

# b) ROI outline
parts = [roi_union] if roi_union.geom_type == "Polygon" else roi_union.geoms
for part in parts:
    xs, ys = part.exterior.xy
    ax.plot(xs, ys, color="black", linewidth=1.2)

# c) Station clusters
cluster_colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
for cl, grp in df_st.groupby("cluster"):
    ax.scatter(
        grp["lon"], grp["lat"],
        marker="x", s=60,
        color=cluster_colors[cl],
        label=f"Cluster {cl}"
    )

# d) Build legend entries
#   Ecoregion patches
eco_patches = [
    Patch(facecolor=fill_colors[nm], edgecolor="grey", alpha=0.5, label=nm)
    for nm in unique_names
]
#   Cluster markers
cluster_handles = [
    Line2D([0], [0], marker="x", color=cluster_colors[cl],
           linestyle="None", markersize=8, label=f"Cluster {cl}")
    for cl in sorted(cluster_colors)
]

# Combine legends: ecoregions first, then clusters
all_handles = eco_patches + cluster_handles

ax.legend(
    handles=all_handles,
    loc="upper left",
    frameon=False,
    ncol=1,
    fontsize="small",
    title="Ecoregions (shaded) and Clusters"
)

stats_txt = (
    f"χ² = {chi2:.2f} (df={dof})\n"
    f"p = {p:.3g}\n"
    f"Cramér’s V = {cramer_v:.3f}"
)
ax.text(
    0.99, 0.99, stats_txt,
    transform=ax.transAxes,
    va="top", ha="right",
    fontsize="small",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    zorder=10
)

# e) Final map settings
minx, miny, maxx, maxy = roi_union.bounds
pad_x, pad_y = (maxx-minx)*0.03, (maxy-miny)*0.03
ax.set_xlim(minx-pad_x, maxx+pad_x)
ax.set_ylim(miny-pad_y, maxy+pad_y)

ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.set_title("Lower Peninsula: Streamflow Clusters & Level III Ecoregions")

ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_out, dpi=300, bbox_inches="tight")
print(f"Map with legend saved to: {fig_out}")

# pack into a dict
row = {
    "cluster_tag": "Ecoregions III",
    "mode":        mode,
    "chi2":        chi2,
    "p_value":     p,
    "dof":         dof,
    "cramer_v":    cramer_v
}

# read existing or start new
if os.path.exists(results_file):
    df_results = pd.read_excel(results_file)
    # concat new row
    df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)
else:
    df_results = pd.DataFrame([row])

# overwrite the sheet with the updated table
df_results.to_excel(results_file, index=False)
