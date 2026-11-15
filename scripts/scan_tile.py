"""Download all years of one VNP46A4 tile, keep the full-resolution layer, and list 0.1-degree cells
with a significant DECREASING Mann-Kendall trend.

Usage: python scripts/scan_tile.py h28v07            (download + scan)
Full tiles are kept as data/tiles/<tile>_<year>.npy (float16 radiance) so any region can be clipped later.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze
import fetch_blackmarble as fb

TILE = sys.argv[1] if len(sys.argv) > 1 else "h28v07"
YEARS = list(range(2012, 2026))
TILES = fb.ROOT / "data" / "tiles"


def get(year, tok):
    out = TILES / f"{TILE}_{year}.npy"
    if out.exists():
        return f"{year}: cached"
    name = fb.tile_file(year, TILE)
    tmp = TILES / f"_{name}"
    t0 = time.time()
    with requests.get(f"{fb.BASE}/{year}/001/{name}", headers={"Authorization": f"Bearer {tok}"},
                      stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    with h5py.File(tmp, "r") as f:
        ds = f[fb.LAYER]
        fill = ds.attrs.get("_FillValue", [65535])[0]
        scale = ds.attrs.get("scale_factor", [0.1])[0]
        raw = ds[:].astype(np.float32)
    np.save(out, np.where(raw == fill, np.nan, raw * scale).astype(np.float16))
    tmp.unlink()
    return f"{year}: {time.time() - t0:.0f}s"


def scan():
    h, v = int(TILE[1:3]), int(TILE[4:6])
    lon0, lat0 = -180 + h * 10, 90 - v * 10
    blocks = np.stack([np.nanmean(np.load(TILES / f"{TILE}_{y}.npy").astype(np.float32)
                                  .reshape(100, 24, 100, 24), axis=(1, 3)) for y in YEARS])
    slope, p = analyze.pixel_trends(blocks, YEARS)
    mean = np.nanmean(blocks, axis=0)
    rows = []
    for i, j in zip(*np.where((p < 0.05) & (slope < 0) & (mean > 2))):
        rows.append((slope[i, j] / mean[i, j] * 100, slope[i, j], p[i, j], mean[i, j],
                     lat0 - (i + .5) * .1, lon0 + (j + .5) * .1))
    rows.sort()
    print(f"{len(rows)} cells with significant decline (mean > 2 nW/cm2/sr)")
    for pct, s, pp, m, la, lo in rows[:60]:
        print(f"{pct:+6.1f}%/yr  slope {s:+7.3f}  p {pp:.3f}  mean {m:6.1f}  lat {la:6.2f} lon {lo:7.2f}")


def clip_region(key):
    """Clip a region from regions.json out of the saved full tiles into data/clip/<key>_<year>.npy."""
    r = analyze.REGIONS[key]
    h, v = int(TILE[1:3]), int(TILE[4:6])
    lon0, lat0, res = -180 + h * 10, 90 - v * 10, 10 / 2400
    c0, c1 = int((r["lon_min"] - lon0) / res), int(np.ceil((r["lon_max"] - lon0) / res))
    r0, r1 = int((lat0 - r["lat_max"]) / res), int(np.ceil((lat0 - r["lat_min"]) / res))
    for y in YEARS:
        a = np.load(TILES / f"{TILE}_{y}.npy")[r0:r1, c0:c1].astype(np.float32)
        np.save(fb.CLIP / f"{key}_{y}.npy", a)
    print(f"clipped {key}: {r1 - r0}x{c1 - c0} px, {len(YEARS)} years")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[2] == "clip":
        clip_region(sys.argv[3]); sys.exit()
    TILES.mkdir(parents=True, exist_ok=True)
    tok = fb.token()
    with ThreadPoolExecutor(max_workers=7) as ex:
        for msg in ex.map(lambda y: get(y, tok), YEARS):
            print(msg, flush=True)
    scan()
