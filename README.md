# Night Light Trend Detective

**Team offByThree · 2026 NASA Space Apps Challenge · Challenge: _Be An Earth System Trend Detective!_**

Live tool: https://khietnn.github.io/night-light-trend-detective/

Night Light Trend Detective helps students and the public investigate whether places are getting brighter or darker at night, **where** the change happens, **how much** it changes per year, and whether the change is **statistically significant**. It uses NASA's **Black Marble** night-light data (VIIRS Day/Night Band, 2012–2025).

## What it does
1. **Select a region**: Ho Chi Minh City, Hanoi, or the Binh Thuan dragon fruit farms, where fields are lit all night to make the plants flower out of season (agricultural light pollution).
2. **Visualise the time series**: annual mean night-time radiance (nW/cm²/sr), 2012–2025.
3. **Quantify the change**: Theil–Sen slope (units per year and % per year).
4. **Test significance**: Mann–Kendall trend test (α = 0.05). Users can change the period and the test re-runs in the browser, which shows how significance depends on record length.
5. **Locate the change**: a per-pixel trend map (Mann–Kendall + Sen's slope for every ~500 m pixel) shows where lights increased or decreased significantly, so opposite trends inside one region become visible.

## How it works
```
NASA LAADS DAAC (VNP46A4 HDF5 tiles)
  └─ scripts/fetch_blackmarble.py   download tiles, clip regions → data/clip/*.npy
       └─ scripts/analyze.py        regional series + per-pixel Mann-Kendall / Theil-Sen
            └─ docs/data/results.json, docs/img/*.png
                 └─ docs/index.html  static web tool (Chart.js, in-browser Mann-Kendall)
```

### Run it yourself
```bash
pip install h5py numpy requests matplotlib scipy pymannkendall
echo EARTHDATA_TOKEN=<your NASA Earthdata token> > .env   # also accept the Black Marble license on LAADS
python scripts/fetch_blackmarble.py      # ~100 MB per tile-year; clipped output is kept, raw files deleted
python scripts/analyze.py
# open docs/index.html via any static server, e.g. python -m http.server -d docs
```

## Data
- NASA Black Marble VNP46A4 Collection 2 (annual night-time lights), layer `AllAngle_Composite_Snow_Free`, 15 arc-second grid, tiles h28v06 and h28v07. Source: NASA LAADS DAAC, https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP46A4/
- Román, M.O. et al. (2018). NASA's Black Marble nighttime lights product suite. Remote Sensing of Environment, 210, 113–143.

## Limitations
- VIIRS DNB is not sensitive to blue light below ~500 nm, so LED conversions can appear as dimming.
- Overpass is ~01:30 local time; it measures upward radiance, not sky glow seen from the ground.
- Annual composites hide seasonal and short events; only monotonic (linear) trends are tested.

## Tools
Python (h5py, NumPy, SciPy, pymannkendall, Matplotlib), HTML/CSS/JavaScript, Chart.js, GitHub Pages.

## Use of Super Intelligence (SI)
Parts of the code (data download, analysis and web page) and documentation text were drafted with an SI assistant (Claude by Anthropic) and reviewed, run and checked by the team. The project idea, region choices and interpretation are the team's own. No SI-generated images are used.

## License
MIT. Data courtesy of NASA; this project is not endorsed by NASA.
