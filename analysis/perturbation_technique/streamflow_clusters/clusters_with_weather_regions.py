import os
import fiona
from shapely.geometry import shape, Point
from shapely.ops import unary_union, transform as shp_transform
from pyproj import CRS, Transformer
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Patch
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency

# ─── CONFIG ────────────────────────────────────────────────────────────────
mode        = "temporal"   # or "spatial", "spatiotemporal"
base_path   = "/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/analysis/perturbation_technique/weather_clusters"
roi_shp     = "/home/talhamuh/water-research/CNN-LSMT/data/raw/Michigan/Final_Michigan_Map/Watershed_Boundary_Intersect_Michigan.shp"
stations_csv= f"/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/analysis/perturbation_technique/streamflow_clusters/{mode}_station_clusters.csv"
results_file = "Clusters_x_regions/cluster_stats.xlsx"

region_paths = [
    f"{base_path}/outputs/kmeans/ppt_4/ppt_4.shp",
    f"{base_path}/outputs/kmeans/tmax_4/tmax_4.shp",
    f"{base_path}/outputs/kmeans/tmin_4/tmin_4.shp",
    f"{base_path}/outputs/kmeans/combined_4/combined_4.shp",
    f"{base_path}/outputs/gmm/ppt_4/ppt_4.shp",
    f"{base_path}/outputs/gmm/tmax_4/tmax_4.shp",
    f"{base_path}/outputs/gmm/tmin_4/tmin_4.shp",
    f"{base_path}/outputs/gmm/combined_4/combined_4.shp",
    f"{base_path}/outputs/fcm/ppt_4/ppt_4.shp",
    f"{base_path}/outputs/fcm/tmax_4/tmax_4.shp",
    f"{base_path}/outputs/fcm/tmin_4/tmin_4.shp",
    f"{base_path}/outputs/fcm/combined_4/combined_4.shp",
]

# ─── LOAD & UNION ROI ──────────────────────────────────────────────────────
roi_parts = []
with fiona.open(roi_shp) as src:
    src_crs = CRS.from_wkt(src.crs_wkt)
    to_wgs  = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True).transform \
              if src_crs.to_epsg()!=4326 else None

    for feat in src:
        g = shape(feat["geometry"])
        if to_wgs:
            g = shp_transform(to_wgs, g)
        roi_parts.append(g)

roi_union = unary_union(roi_parts)

# ─── LOOP OVER EVERY CLUSTER SHAPEFILE ─────────────────────────────────────
for shp_path in region_paths:
    # infer variable & technique from path
    parts       = shp_path.split(os.sep)
    technique   = parts[-3]          # "kmeans", "gmm", or "fcm"
    var_dir     = parts[-2]          # "ppt_4", "tmax_4", etc.
    cluster_tag = f"{var_dir}_{technique}"

    # prepare output filenames
    table_out = f"Clusters_x_regions/{mode}_cluster_by_{cluster_tag}.csv"
    fig_out   = f"Clusters_x_regions/{mode}_clusters_vs_{cluster_tag}.png"

    # 1) load & clip this cluster‐polygon shapefile
    cluster_geoms, cluster_ids = [], []
    with fiona.open(shp_path) as src:
        crs_in = CRS.from_wkt(src.crs_wkt)
        to_wgs = Transformer.from_crs(crs_in, "EPSG:4326", always_xy=True).transform \
                  if crs_in.to_epsg()!=4326 else None

        for feat in src:
            geom = shape(feat["geometry"])
            if to_wgs:
                geom = shp_transform(to_wgs, geom)
            clipped = geom.intersection(roi_union)
            if clipped.is_empty:
                continue
            cluster_geoms.append(clipped)
            cluster_ids.append(int(round(feat["properties"]["cluster_id"])))

    # 2) read stations & assign into these polygons
    df_st = pd.read_csv(stations_csv)
    records = []
    for _, row in df_st.iterrows():
        pt = Point(row.lon, row.lat)
        if not roi_union.contains(pt):
            continue
        cid = next((cid for g, cid in zip(cluster_geoms, cluster_ids) if g.contains(pt)), None)
        records.append({
            "station_cluster": int(row.cluster),
            "ppt_cluster": cid,
            "lon": row.lon,
            "lat": row.lat
        })
    df_pairs = pd.DataFrame(records)

    # 3) contingency + stats
    table = pd.crosstab(df_pairs.station_cluster, df_pairs.ppt_cluster)
    table.to_csv(table_out)

    chi2, p, dof, _ = chi2_contingency(table.fillna(0))
    N = table.values.sum()
    cramer_v = (chi2/(N*(min(table.shape)-1)))**0.5

    print(f"\n=== {cluster_tag} ===")
    print(f"χ²={chi2:.2f} (df={dof}), p={p:.3g}, Cramér’s V={cramer_v:.3f}")
    print(f"saved: {table_out}")

    # 4) plot
    fig, ax = plt.subplots(figsize=(8,10))

    # pastel fill polygons
    cmap   = plt.get_cmap("tab20", len(cluster_ids))
    colors = {cid: cmap(i) for i, cid in enumerate(cluster_ids)}
    for geom, cid in zip(cluster_geoms, cluster_ids):
        if geom.geom_type=="Polygon":
            xs, ys = geom.exterior.xy
            ax.add_patch(MplPolygon(
                list(zip(xs, ys)),
                facecolor=colors[cid],
                edgecolor="grey",
                linewidth=0.5,
                alpha=0.3
            ))
        else:
            for part in geom.geoms:
                xs, ys = part.exterior.xy
                ax.add_patch(MplPolygon(
                    list(zip(xs, ys)),
                    facecolor=colors[cid],
                    edgecolor="grey",
                    linewidth=0.5,
                    alpha=0.3
                ))

    # ROI outline
    outline = [roi_union] if roi_union.geom_type=="Polygon" else roi_union.geoms
    for part in outline:
        xs, ys = part.exterior.xy
        ax.plot(xs, ys, "k-", linewidth=1.2)

    # station markers
    sc_colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    for sc, grp in df_pairs.groupby("station_cluster"):
        ax.scatter(
            grp.lon, grp.lat,
            marker="x", s=60,
            color=sc_colors.get(sc, "k")
        )

    # ─── DEDUPED LEGEND ───────────────────────────────────────────────────────
    # unique PPT clusters
    unique_ppt_ids = sorted(set(cluster_ids))
    ppt_patches = [
        Patch(facecolor=colors[cid], edgecolor="grey", alpha=0.5, label=f"PPT Cl {cid}")
        for cid in unique_ppt_ids
    ]

    # unique station clusters
    unique_sc = sorted(df_pairs["station_cluster"].unique())
    station_handles = [
        Line2D([0], [0],
               marker="x",
               color=sc_colors[sc],
               linestyle="None",
               markersize=8,
               label=f"Station Cl {sc}")
        for sc in unique_sc
    ]

    ax.legend(
        handles=ppt_patches + station_handles,
        loc="upper left",
        frameon=False,
        fontsize="small",
        title="Clusters"
    )

    # stats box
    stats_txt = f"χ²={chi2:.2f} (df={dof})\np={p:.3g}\nV={cramer_v:.3f}"
    ax.text(
        0.99, 0.99, stats_txt,
        transform=ax.transAxes,
        va="top", ha="right",
        fontsize="small",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    # final touches
    minx, miny, maxx, maxy = roi_union.bounds
    dx, dy = (maxx-minx)*0.03, (maxy-miny)*0.03
    ax.set_xlim(minx-dx, maxx+dx)
    ax.set_ylim(miny-dy, maxy+dy)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"{mode}: {cluster_tag}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {fig_out}")
    
    
    # pack into a dict
    row = {
        "cluster_tag": cluster_tag,
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

