"""Dark-theme figures for the team slide deck, made only from docs/data/results.json and the clipped NASA data."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"
R = json.loads((ROOT / "docs" / "data" / "results.json").read_text(encoding="utf-8"))["regions"]
BG, FG, MUTED, GRID = "#0f1729", "#e8ecf8", "#9aa5c4", "#26314f"
AMBER, RED, BLUE = "#f5b942", "#ff6b5b", "#4aa3ff"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "text.color": FG,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11})


def sen_line(years, values, slope):
    inter = np.median([v - slope * y for y, v in zip(years, values)])
    return [inter + slope * y for y in years]


def verdict(v):
    if not v["significant"]:
        return f"no significant trend (p = {v['p_value']:.2f})"
    p = "p < 0.001" if v["p_value"] < 0.001 else f"p = {v['p_value']:.3f}"
    return f"{v['percent_change_per_year']:+.1f}%/yr, {p}"


def regions_panel():
    fig, axes = plt.subplots(3, 1, figsize=(6, 6.4), dpi=170, sharex=True)
    for ax, key in zip(axes, ["hcmc", "hanoi", "binhthuan"]):
        r = R[key]; v = r["variables"]["mean_radiance"]; y = r["years"]
        ax.plot(y, v["values"], "o-", color=AMBER, ms=3.5, lw=1.8)
        ax.plot(y, sen_line(y, v["values"], v["sen_slope_per_year"]), "--", color=MUTED, lw=1.2)
        ax.set_title(f"{r['label']}: {verdict(v)}", loc="left", fontsize=11, color=FG)
        ax.grid(color=GRID, lw=.6); ax.tick_params(labelsize=9)
    axes[1].set_ylabel("Mean night radiance (nW/cm²/sr)")
    fig.text(.99, .005, "Data: NASA Black Marble VNP46A4 · dashed = Theil–Sen trend", ha="right", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, .02, 1, 1)); fig.savefig(OUT / "slide_regions.png", facecolor=BG); plt.close(fig)


def opposite_panel():
    key = "binhthuan"; reg = analyze.REGIONS[key]; r = R[key]; years = r["years"]
    stack = np.stack([np.load(analyze.CLIP / f"{key}_{yy}.npy") for yy in years])
    slope, p = analyze.pixel_trends(stack, years)
    fig = plt.figure(figsize=(12, 4), dpi=150)
    ax1 = fig.add_axes([0.04, 0.12, 0.30, 0.76]); ax2 = fig.add_axes([0.42, 0.14, 0.56, 0.72])
    ext = [reg["lon_min"], reg["lon_max"], reg["lat_min"], reg["lat_max"]]
    lim = np.nanpercentile(np.abs(slope), 98)
    ax1.imshow(np.where(~np.isnan(p), 0.35, np.nan), cmap="Greys", vmin=0, vmax=1, extent=ext)
    ax1.imshow(np.where(p < 0.05, slope, np.nan), cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext)
    ax1.set_title("Binh Thuan pixels 2012–2025\nred = brighter, blue = dimmer, grey = no trend", fontsize=10)
    ax1.tick_params(labelsize=8)
    c, f = r["variables"]["core"], r["variables"]["fringe"]
    ax2.plot(years, np.array(c["values"]) / c["values"][0] * 100, "o-", color=BLUE, lw=2,
             label=f"Brightest farm areas of 2012: {verdict(c)}")
    ax2.plot(years, np.array(f["values"]) / f["values"][0] * 100, "o-", color=RED, lw=2,
             label=f"Areas that were dark in 2012: {verdict(f)}")
    ax2.axhline(100, color=MUTED, lw=.8, ls=":")
    ax2.set_ylabel("Brightness index (2012 = 100)"); ax2.grid(color=GRID, lw=.6)
    ax2.set_title("Opposite trends inside one region (Mann–Kendall)", fontsize=11)
    ax2.legend(facecolor=BG, edgecolor=GRID, fontsize=9, loc="center right", bbox_to_anchor=(1, 0.62))
    fig.text(.99, .01, "Data: NASA Black Marble VNP46A4 · team analysis", ha="right", fontsize=7.5, color=MUTED)
    fig.savefig(OUT / "slide_opposite.png", facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    regions_panel(); opposite_panel(); print("ok")
