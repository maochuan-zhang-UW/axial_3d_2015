"""
Prepare data for a 3D interactive visualization (Plotly.js) of Axial
Seamount for the Jan22-Apr25 2015 window used throughout this project,
in the style of https://github.com/MaleenKidiwela/axial_visuals -
caldera-wall fault point clouds + AMC roof surface + caldera rim +
earthquake catalog, all in one local km frame.

Reuses that repo's own AXCC1-centered coordinate convention (same
lat0/lon0/km-per-degree as their latlon2xy_no_rotate.m) so its fault-wall
data (downloaded separately, not redistributed here) lines up with our
own catalog/detections without reprojecting their data.

Depth convention: km, POSITIVE DOWN, for every layer in the output.

Usage:
    python prepare_3d_data.py \
        --west-wall <path to I_fitting_WW_slices.mat> \
        --east-wall <path to East_Felix_06_Dis.mat> \
        --relocated ../../02-data/catalogs/relocated_2015-04-23_t20.csv \
                    ../../02-data/catalogs/relocated_2015-04-24_t20.csv \
                    ../../02-data/catalogs/relocated_2015-04-25_t20.csv \
        --catalog-start 2015-04-23 --catalog-end 2015-04-26 \
        --out data_3d.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

import catalog_io
from magma_chamber import MagmaChamber
from plot_relocated import CALDERA_RIM, load

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_OFFSET_M = 2000.0

# Origin/scale matching axial_visuals' latlon2xy_no_rotate.m exactly, so
# the borrowed fault-wall data (already in this frame) lines up with our
# own catalog without reprojecting it.
LAT0, LON0 = 45.9547, -130.0089
KM_PER_DEG_LAT = 111.19


def to_local(lat, lon):
    km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.radians(LAT0))
    x = (np.asarray(lon) - LON0) * km_per_deg_lon
    y = (np.asarray(lat) - LAT0) * KM_PER_DEG_LAT
    return x, y


def local_to_latlon(x, y):
    km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.radians(LAT0))
    lon = LON0 + np.asarray(x) / km_per_deg_lon
    lat = LAT0 + np.asarray(y) / KM_PER_DEG_LAT
    return lat, lon


def load_west_wall(path):
    d = sio.loadmat(path)
    a = d["fit_x_y_z"]
    return {"x": a[:, 0].tolist(), "y": a[:, 1].tolist(),
           "z": (-a[:, 2]).tolist()}  # their z is negative-down -> flip


def load_east_wall(path):
    d = sio.loadmat(path)
    felix = d["Felix"][0]
    lon = np.array([f["lon"].item() if f["lon"].size else np.nan
                    for f in felix])
    lat = np.array([f["lat"].item() if f["lat"].size else np.nan
                    for f in felix])
    depth = np.array([f["depth"].item() if f["depth"].size else np.nan
                      for f in felix])
    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(depth)
    x, y = to_local(lat[ok], lon[ok])
    return {"x": x.tolist(), "y": y.tolist(), "z": depth[ok].tolist()}


def amc_surface(nx=60, ny=70):
    amc = MagmaChamber()
    x_edges = np.linspace(-3.5, 2.0, nx)
    y_edges = np.linspace(-4.5, 2.5, ny)
    X, Y = np.meshgrid(x_edges, y_edges)
    lat_q, lon_q = local_to_latlon(X.ravel(), Y.ravel())
    z = amc.roof_depth_local(lon_q, lat_q).reshape(X.shape)
    # NaN outside the imaged AMC extent -> JSON null (NaN isn't valid JSON
    # and breaks JSON.parse in the browser; Plotly Surface treats null as
    # a gap, which is exactly what "outside the imaged extent" means).
    z_list = [[None if np.isnan(v) else v for v in row] for row in z]
    return {"x": x_edges.tolist(), "y": y_edges.tolist(), "z": z_list}


def caldera_rim_local():
    lon = np.array([p[0] for p in CALDERA_RIM])
    lat = np.array([p[1] for p in CALDERA_RIM])
    x, y = to_local(lat, lon)
    return {"x": x.tolist(), "y": y.tolist()}


def load_catalog_and_detections(args):
    events_df, picks_df, mag, reloc = catalog_io.load_all(args.data_dir)
    ev = events_df.copy()
    ev["time"] = pd.to_datetime(
        ev["time"].apply(lambda t: t.isoformat()), utc=True, format="ISO8601")
    ev = ev[(ev["time"] >= pd.Timestamp(args.catalog_start, tz="UTC"))
           & (ev["time"] < pd.Timestamp(args.catalog_end, tz="UTC"))]
    loc = reloc.reindex(ev.index)
    lat = loc["latitude"].to_numpy().copy()
    lon = loc["longitude"].to_numpy().copy()
    depth = loc["depth_km"].to_numpy().copy()
    missing = pd.isna(lat)
    lat[missing] = ev["latitude"].to_numpy()[missing]
    lon[missing] = ev["longitude"].to_numpy()[missing]
    depth[missing] = ev["depth_km"].to_numpy()[missing]
    cat = pd.DataFrame({"latitude": lat, "longitude": lon, "depth_km": depth,
                        "time": ev["time"].to_numpy()})
    cat = cat[cat["latitude"].between(45.85, 46.05)
             & cat["longitude"].between(-130.15, -129.90)
             & cat["depth_km"].between(-3, 6)]
    cx, cy = to_local(cat["latitude"].to_numpy(), cat["longitude"].to_numpy())
    cat_out = {
        "x": cx.tolist(), "y": cy.tolist(),
        "z": cat["depth_km"].tolist(),
        "day": cat["time"].dt.strftime("%Y-%m-%d").tolist(),
        "time": cat["time"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
    }

    det = load(args.relocated)
    det["detect_time"] = pd.to_datetime(det["detect_time"], utc=True)
    horiz_m = np.hypot(det["dx_m"].fillna(0), det["dy_m"].fillna(0))
    is_outlier = (det["location_source"] == "relative") & (
        (horiz_m > MAX_OFFSET_M) | (det["dz_m"].abs() > MAX_OFFSET_M))
    det = det[~is_outlier].dropna(subset=["latitude", "longitude", "depth_km"])
    det = det[(det["detect_time"] >= pd.Timestamp(args.catalog_start, tz="UTC"))
             & (det["detect_time"] < pd.Timestamp(args.catalog_end, tz="UTC"))]
    dx, dy = to_local(det["latitude"].to_numpy(), det["longitude"].to_numpy())
    det_out = {
        "x": dx.tolist(), "y": dy.tolist(),
        "z": det["depth_km"].tolist(),
        "day": det["detect_time"].dt.strftime("%Y-%m-%d").tolist(),
        "time": det["detect_time"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
    }
    print(f"catalog: {len(cat_out['x'])} events, "
         f"new detections: {len(det_out['x'])} events")
    return cat_out, det_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--west-wall", required=True)
    parser.add_argument("--east-wall", required=True)
    parser.add_argument("--relocated", nargs="+", required=True)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "02-data"))
    parser.add_argument("--catalog-start", required=True)
    parser.add_argument("--catalog-end", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cat, det = load_catalog_and_detections(args)

    out = {
        "origin": {"lat0": LAT0, "lon0": LON0},
        "west_wall": load_west_wall(args.west_wall),
        "east_wall": load_east_wall(args.east_wall),
        "amc": amc_surface(),
        "caldera_rim": caldera_rim_local(),
        "catalog": cat,
        "new_detections": det,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, allow_nan=False)
    print(f"Saved to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
