"""Download NASA Black Marble VNP46A4 (annual night lights) tiles and clip study regions.

Reads EARTHDATA_TOKEN from ../.env. Raw HDF5 tiles are deleted after clipping.
Output: data/clip/<region>_<year>.npy (float32 radiance in nW/cm^2/sr, NaN = no data)
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLIP = ROOT / "data" / "clip"
BASE = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP46A4"
LAYER = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/AllAngle_Composite_Snow_Free"

# name: (tile, lon_min, lat_min, lon_max, lat_max)
REGIONS = json.loads((ROOT / "scripts" / "regions.json").read_text(encoding="utf-8"))


def token():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("EARTHDATA_TOKEN="):
            return line.split("=", 1)[1].strip()
    sys.exit("EARTHDATA_TOKEN missing in .env")


def tile_file(year, tile):
    listing = requests.get(f"{BASE}/{year}/001.json", timeout=60).json()
    listing = listing.get("content", listing)
    for item in listing:
        if f".{tile}." in item["name"]:
            return item["name"]
    return None


def clip(h5_path, tile, regions):
    h, v = int(tile[1:3]), int(tile[4:6])
    lon0, lat0, res = -180 + h * 10, 90 - v * 10, 10 / 2400
    with h5py.File(h5_path, "r") as f:
        ds = f[LAYER]
        fill = ds.attrs.get("_FillValue", [65535])[0]
        scale = ds.attrs.get("scale_factor", [0.1])[0]
        offset = ds.attrs.get("add_offset", [0.0])[0]
        out = {}
        for name, r in regions.items():
            c0, c1 = int((r["lon_min"] - lon0) / res), int(np.ceil((r["lon_max"] - lon0) / res))
            r0, r1 = int((lat0 - r["lat_max"]) / res), int(np.ceil((lat0 - r["lat_min"]) / res))
            raw = ds[r0:r1, c0:c1].astype(np.float32)
            arr = np.where(raw == fill, np.nan, raw * scale + offset).astype(np.float32)
            out[name] = arr
    return out


def job(year, tile, regions, tok):
    todo = {n: r for n, r in regions.items() if not (CLIP / f"{n}_{year}.npy").exists()}
    if not todo:
        return f"{year} {tile}: cached"
    name = tile_file(year, tile)
    if not name:
        return f"{year} {tile}: NOT FOUND"
    dest = RAW / name
    t0 = time.time()
    with requests.get(f"{BASE}/{year}/001/{name}", headers={"Authorization": f"Bearer {tok}"},
                      stream=True, timeout=120) as resp:
        resp.raise_for_status()
        if "html" in resp.headers.get("Content-Type", ""):
            raise RuntimeError("Got HTML instead of data - check the Earthdata token")
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    mb = dest.stat().st_size / 1e6
    for n, arr in clip(dest, tile, todo).items():
        np.save(CLIP / f"{n}_{year}.npy", arr)
    dest.unlink()
    return f"{year} {tile}: {mb:.0f} MB in {time.time() - t0:.0f}s"


def main():
    years = [int(a) for a in sys.argv[1:]] or list(range(2012, 2026))
    RAW.mkdir(parents=True, exist_ok=True)
    CLIP.mkdir(parents=True, exist_ok=True)
    tok = token()
    by_tile = {}
    for n, r in REGIONS.items():
        by_tile.setdefault(r["tile"], {})[n] = r
    jobs = [(y, t, regs) for t, regs in by_tile.items() for y in years]  # region order = priority
    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", 4))) as ex:
        futs = [ex.submit(job, y, t, regs, tok) for y, t, regs in jobs]
        for fut in as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception as e:  # keep other downloads going
                print("ERROR", e, flush=True)


if __name__ == "__main__":
    main()
