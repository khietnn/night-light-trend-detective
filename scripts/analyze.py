"""Trend analysis of NASA Black Marble annual night lights for each study region.

Input : data/clip/<region>_<year>.npy  (from fetch_blackmarble.py)
Output: docs/data/results.json, docs/img/<region>_series.png, docs/img/<region>_trendmap.png

Statistics: Mann-Kendall test (significance) + Theil-Sen slope (rate of change).
Code written with assistance from an SI tool (Claude); reviewed by team offByThree.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymannkendall as mk
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "data" / "clip"
OUT = ROOT / "docs"
REGIONS = json.loads((ROOT / "scripts" / "regions.json").read_text(encoding="utf-8"))
ALPHA = 0.05
UNIT = "nW/cm²/sr"
LIT = 5.0  # radiance threshold for a "lit" pixel


def pixel_trends(stack, years):
    """Vectorised Mann-Kendall p-value and Theil-Sen slope for every pixel."""
    n = len(years)
    i, j = np.triu_indices(n, 1)
    diff = stack[j] - stack[i]                      # (pairs, H, W)
    s = np.sign(diff).sum(axis=0)
    var = n * (n - 1) * (2 * n + 5) / 18            # no tie correction
    z = np.where(s > 0, (s - 1) / np.sqrt(var), np.where(s < 0, (s + 1) / np.sqrt(var), 0))
    p = 2 * (1 - norm.cdf(np.abs(z)))
    dt = (np.array(years)[j] - np.array(years)[i])[:, None, None]
    slope = np.median(diff / dt, axis=0)
    bad = np.isnan(stack).any(axis=0)
    p[bad], slope[bad] = np.nan, np.nan
    return slope, p


def main():
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    results = {"unit": UNIT, "alpha": ALPHA, "regions": {}}

    for key, reg in REGIONS.items():
        years = sorted(int(p.stem.split("_")[-1]) for p in CLIP.glob(f"{key}_*.npy"))
        if len(years) < 14:  # only publish regions with the full 2012-2025 record
            print(f"{key}: only {len(years)} years, skipped")
            continue
        stack = np.stack([np.load(CLIP / f"{key}_{y}.npy") for y in years])
        series = [float(np.nanmean(a)) for a in stack]

        # Selectable variables derived from the same NASA layer
        base = stack[0]
        core = base >= np.nanpercentile(base, 90)      # already bright in first year
        fringe = base <= np.nanpercentile(base, 50)    # dark in first year
        variables = {
            "mean_radiance": ("Average night-time brightness", UNIT, series,
                              "Mean radiance of all pixels in the region."),
            "lit_area": ("Lit area", "% of region",
                         [float(np.nanmean(a >= LIT) * 100) for a in stack],
                         f"Share of pixels brighter than {LIT} {UNIT}: a proxy for how far lights spread (urban sprawl)."),
            "core": ("Brightness of established bright areas", UNIT,
                     [float(np.nanmean(a[core])) for a in stack],
                     f"Mean radiance of pixels that were already in the brightest 10% in {years[0]} (old centres, ports, industry)."),
            "fringe": ("Brightness of areas that were dark", UNIT,
                       [float(np.nanmean(a[fringe])) for a in stack],
                       f"Mean radiance of pixels that were in the darkest 50% in {years[0]} (rural land, new suburbs, sea)."),
        }
        var_out = {}
        for vk, (label, unit, vals, desc) in variables.items():
            vt = mk.original_test(vals, alpha=ALPHA)
            var_out[vk] = {"label": label, "unit": unit, "desc": desc, "values": [round(v, 4) for v in vals],
                           "trend": vt.trend, "p_value": float(vt.p), "significant": bool(vt.h),
                           "sen_slope_per_year": float(vt.slope),
                           "percent_change_per_year": float(vt.slope / np.mean(vals) * 100) if np.mean(vals) else 0.0}

        t = mk.original_test(series, alpha=ALPHA)
        slope, p = pixel_trends(stack, years)
        valid = ~np.isnan(p)
        up = (p < ALPHA) & (slope > 0)
        down = (p < ALPHA) & (slope < 0)
        results["regions"][key] = {
            "label": reg["label"],
            "bbox": [reg["lon_min"], reg["lat_min"], reg["lon_max"], reg["lat_max"]],
            "years": years,
            "mean_radiance": [round(v, 3) for v in series],
            "trend": t.trend,
            "p_value": float(t.p),
            "significant": bool(t.h),
            "sen_slope_per_year": float(t.slope),
            "percent_change_per_year": float(t.slope / np.mean(series) * 100),
            "pixels_brightening_pct": float(up.sum() / valid.sum() * 100),
            "pixels_dimming_pct": float(down.sum() / valid.sum() * 100),
            "variables": var_out,
        }

        # time series + Sen's line
        fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
        ax.plot(years, series, "o-", color="#f5a623", label="Mean night-time radiance")
        ax.plot(years, [t.intercept + t.slope * k for k in range(len(years))], "--", color="#333",
                label=f"Sen's slope {t.slope:+.3f} {UNIT}/yr")
        verdict = "significant" if t.h else "not significant"
        ax.set_title(f"{reg['label']}: {t.trend} (Mann-Kendall p = {t.p:.3g}, {verdict})")
        ax.set_xlabel("Year"); ax.set_ylabel(f"Radiance ({UNIT})")
        ax.grid(alpha=.3); ax.legend()
        fig.text(.99, .01, "Data: NASA Black Marble VNP46A4", ha="right", fontsize=7, color="#666")
        fig.tight_layout(); fig.savefig(OUT / "img" / f"{key}_series.png"); plt.close(fig)

        # per-pixel trend map: colour = slope where significant, grey elsewhere
        lim = np.nanpercentile(np.abs(slope), 98) or 1
        ext = [reg["lon_min"], reg["lon_max"], reg["lat_min"], reg["lat_max"]]
        fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
        ax.imshow(np.where(valid, 0.5, np.nan), cmap="Greys", vmin=0, vmax=1.5, extent=ext)
        im = ax.imshow(np.where(p < ALPHA, slope, np.nan), cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext)
        fig.colorbar(im, label=f"Sen's slope ({UNIT} per year)")
        ax.set_title(f"{reg['label']}: where night lights changed significantly\n"
                     f"(red = brighter, blue = dimmer, grey = no significant trend, p < {ALPHA})", fontsize=9)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        fig.tight_layout(); fig.savefig(OUT / "img" / f"{key}_trendmap.png"); plt.close(fig)

        r = results["regions"][key]
        print(f"{reg['label']}: {r['trend']}, p={r['p_value']:.4g}, slope={r['sen_slope_per_year']:+.4f}/yr "
              f"({r['percent_change_per_year']:+.2f}%/yr), up {r['pixels_brightening_pct']:.1f}% "
              f"down {r['pixels_dimming_pct']:.1f}% of pixels, n={len(years)}")
        for vk, v in var_out.items():
            print(f"   {vk:14s} {v['trend']:12s} p={v['p_value']:.3g} slope={v['sen_slope_per_year']:+.4f} "
                  f"({v['percent_change_per_year']:+.2f}%/yr) first={v['values'][0]:.2f} last={v['values'][-1]:.2f}")

    (OUT / "data" / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
