# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Part 7: Comprehensive multi-grid comparison on *Quercus suber*
#
# Notebook 02 compared a regular 1° lat-lon grid to HEALPix on the same
# Q. suber GBIF data. That comparison is methodologically clean but a
# strawman — biodiversity practitioners do not actually use raw lat-lon
# as a serious aggregation grid. They use **equal-area projections**
# (Behrmann, Mollweide), **the EEA reference grid** (LAEA Europe,
# the INSPIRE / Habitats Directive standard for European biodiversity
# reporting), or modern **DGGS** (HEALPix, H3, ISEA3H, rHEALPix).
#
# This notebook applies the same Q. suber dataset to **every grid the
# biodiversity community might pick** and asks: do the equal-area
# alternatives all agree? Where do the count-bias and AI-readiness
# arguments separate them?
#
# This is the iteration that broadens notebook 02's two-grid story
# into a comprehensive comparison. Grids are added in this commit and
# a series of follow-ups:
#
# - This commit: **lat-lon 1°**, **HEALPix nside=64**, **H3 res 3**
# - Follow-up: + rHEALPix, Mollweide, EEA reference grid (LAEA Europe)
# - Follow-up: + ISEA3H (requires the Docker container with DGGRID v8.41)
#
# ## Environment requirement
#
# The new dependency in this notebook is `h3` (Uber H3 v4). Update your
# environment with `mamba env update -f environment.yml` or run inside
# the Docker container at `ghcr.io/annefou/dggs-biodiversity-bias:main`
# where it is pre-installed.

# %%
import json
from collections import Counter
from pathlib import Path

import cartopy.crs as ccrs
import h3
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Step 1: Load *Quercus suber* occurrences from the local cache
#
# Notebook 02 downloaded ~20,100 georeferenced records from GBIF
# (taxon key 2879411) and cached them as `data/quercus_suber_gbif.json`.
# We re-use that cache here so this notebook is offline-safe.

# %%
DATA_DIR = Path("../data")
CACHE_PATH = DATA_DIR / "quercus_suber_gbif.json"
if not CACHE_PATH.exists():
    raise FileNotFoundError(
        f"Expected cached GBIF data at {CACHE_PATH}. "
        "Run notebook 02 first to populate the cache."
    )

with CACHE_PATH.open() as f:
    coords = json.load(f)

lats = np.array([c[0] for c in coords])
lons = np.array([c[1] for c in coords])
print(f"Loaded {len(coords):,} occurrences from cache")
print(f"Latitude range:  {lats.min():.2f}°  to  {lats.max():.2f}°")
print(f"Longitude range: {lons.min():.2f}°  to  {lons.max():.2f}°")

# %% [markdown]
# ## Step 2: Clip to *Q. suber's* Mediterranean range
#
# Same window as notebook 02 — the western Mediterranean basin where
# Q. suber actually grows.

# %%
LAT_MIN, LAT_MAX = 30.0, 46.0
LON_MIN, LON_MAX = -10.0, 25.0

in_range = (
    (lats >= LAT_MIN) & (lats <= LAT_MAX)
    & (lons >= LON_MIN) & (lons <= LON_MAX)
)
lats_r = lats[in_range]
lons_r = lons[in_range]
print(f"Records inside Mediterranean window: {len(lats_r):,} of {len(lats):,}")

# %% [markdown]
# ## Step 3: Aggregate on a 1° lat-lon grid (cautionary baseline)
#
# Lat-lon is included as a cautionary baseline rather than a serious
# aggregation choice. At 30°N a 1° cell covers ~12,300 km²; at 46°N
# it covers ~8,600 km². Same nominal "1° cell", ~43% area difference
# — pure grid geometry, no ecological content.

# %%
GRID_RES = 1.0
lat_edges = np.arange(LAT_MIN, LAT_MAX + GRID_RES, GRID_RES)
lon_edges = np.arange(LON_MIN, LON_MAX + GRID_RES, GRID_RES)

counts_latlon, _, _ = np.histogram2d(lats_r, lons_r, bins=[lat_edges, lon_edges])

R_KM = 6371.0
cell_area_latlon = np.zeros_like(counts_latlon)
for i, (lat_s, lat_n) in enumerate(zip(lat_edges[:-1], lat_edges[1:])):
    cell_area_latlon[i, :] = (
        2 * np.pi * R_KM**2
        * abs(np.sin(np.radians(lat_n)) - np.sin(np.radians(lat_s)))
        * (GRID_RES / 360)
    )

density_latlon = np.zeros_like(counts_latlon)
nz = counts_latlon > 0
density_latlon[nz] = counts_latlon[nz] / cell_area_latlon[nz]
print(f"Lat-lon 1°: {counts_latlon.shape[0]} × {counts_latlon.shape[1]} cells, "
      f"{int(np.sum(nz)):,} occupied")

# %% [markdown]
# ## Step 4: Aggregate on HEALPix nside=64 (equal-area DGGS)
#
# HEALPix is the equal-area DGGS we are advocating. At nside=64 every
# cell on the sphere has identical area (~10,400 km²).

# %%
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
HEALPIX_CELL_AREA = 4 * np.pi * R_KM**2 / NPIX

theta = np.radians(90.0 - lats_r)
phi = np.radians(lons_r % 360)
pix = hp.ang2pix(NSIDE, theta, phi)
counts_healpix = np.bincount(pix, minlength=NPIX)
print(f"HEALPix nside={NSIDE}: {NPIX:,} cells globally, "
      f"each {HEALPIX_CELL_AREA:,.0f} km²; "
      f"{int(np.sum(counts_healpix > 0)):,} occupied in window")

# %% [markdown]
# ## Step 5: Aggregate on H3 resolution 3 (icosahedral hexagonal DGGS)
#
# H3 is Uber's open-source hexagonal DGGS. Cells are nearly equal
# area (~1.4% variation across the sphere — vs HEALPix's exact
# zero variation, vs lat-lon's >100% variation across latitudes).
# H3 res 3 has cells of ~12,400 km² average area, comparable to
# HEALPix nside=64 (~10,400 km²).

# %%
H3_RES = 3

h3_cells = [h3.latlng_to_cell(float(lat), float(lon), H3_RES)
            for lat, lon in zip(lats_r, lons_r)]
counts_h3 = Counter(h3_cells)

# Per-cell area (H3 cells vary ~1.4% so we compute exact area per cell)
h3_cell_area = {c: h3.cell_area(c, unit="km^2") for c in counts_h3}
mean_h3_area = float(np.mean(list(h3_cell_area.values()))) if h3_cell_area else 0.0
print(f"H3 res {H3_RES}: {len(counts_h3):,} occupied cells, "
      f"mean cell area {mean_h3_area:,.0f} km²")

# %% [markdown]
# ## Step 6: Build a common colour scale and rasterise each grid
#
# To compare the three grids fairly we render each on the same plotting
# mesh, dividing counts by per-cell area so the colour scale is
# **records per km²** in every panel. This makes equal-area cells
# comparable to lat-lon's variable-area cells without per-grid
# rescaling.

# %%
plot_lons = np.linspace(LON_MIN, LON_MAX, 500)
plot_lats = np.linspace(LAT_MIN, LAT_MAX, 300)
plot_lon_g, plot_lat_g = np.meshgrid(plot_lons, plot_lats)

# HEALPix: lookup pix per plot pixel
plot_theta = np.radians(90.0 - plot_lat_g)
plot_phi = np.radians(plot_lon_g % 360)
plot_pix = hp.ang2pix(NSIDE, plot_theta, plot_phi)
healpix_field = counts_healpix[plot_pix].astype(float)
healpix_density = healpix_field / HEALPIX_CELL_AREA
healpix_density_masked = np.ma.masked_where(healpix_field == 0, healpix_density)

# H3: lookup cell per plot pixel
plot_cells = [
    h3.latlng_to_cell(float(lat), float(lon), H3_RES)
    for lat, lon in zip(plot_lat_g.ravel(), plot_lon_g.ravel())
]
plot_counts_h3 = np.array([counts_h3.get(c, 0) for c in plot_cells], dtype=float)
plot_areas_h3 = np.array([h3_cell_area.get(c, mean_h3_area) for c in plot_cells])
h3_density = (plot_counts_h3 / plot_areas_h3).reshape(plot_lat_g.shape)
h3_density_masked = np.ma.masked_where(plot_counts_h3.reshape(plot_lat_g.shape) == 0,
                                       h3_density)

# Lat-lon: already on its own grid
density_latlon_masked = np.ma.masked_where(counts_latlon == 0, density_latlon)

shared_vmax = float(np.percentile(np.concatenate([
    density_latlon_masked.compressed(),
    healpix_density_masked.compressed(),
    h3_density_masked.compressed(),
]), 98))

# %% [markdown]
# ## Step 7: Three-panel comparison
#
# Same data, three grids. The two equal-area grids (HEALPix and H3)
# should agree on the apparent density pattern. The lat-lon panel
# is the cautionary case.

# %%
fig = plt.figure(figsize=(18, 6))
gs = fig.add_gridspec(1, 3, wspace=0.18)

panels = [
    ("A. Lat-lon 1° (cautionary baseline)\nrecords / km²",
     density_latlon_masked, lon_edges, lat_edges, "edges"),
    (f"B. HEALPix nside={NSIDE} (equal-area DGGS)\nrecords / km²",
     healpix_density_masked, plot_lons, plot_lats, "centres"),
    (f"C. H3 resolution {H3_RES} (hexagonal DGGS, ~equal-area)\nrecords / km²",
     h3_density_masked, plot_lons, plot_lats, "centres"),
]

for k, (title, data, x, y, kind) in enumerate(panels):
    ax = fig.add_subplot(gs[0, k], projection=ccrs.PlateCarree())
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    if kind == "edges":
        im = ax.pcolormesh(x, y, data, transform=ccrs.PlateCarree(),
                           cmap="viridis", vmin=0, vmax=shared_vmax)
    else:
        im = ax.pcolormesh(x, y, data, transform=ccrs.PlateCarree(),
                           cmap="viridis", vmin=0, vmax=shared_vmax)
    ax.coastlines(linewidth=0.6, color="grey")
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.08,
                 label="Records per km²", shrink=0.9)

fig.suptitle(
    "Quercus suber GBIF — same data on three grids "
    "(lat-lon vs HEALPix vs H3)",
    fontsize=14, fontweight="bold", y=1.02,
)
plt.savefig("../images/multigrid_quercus_suber.png", dpi=150,
            bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Conclusion (this iteration)
#
# Two equal-area DGGS — HEALPix (exactly equal-area) and H3 (~1.4%
# variation) — produce nearly identical density patterns over Q. suber's
# Mediterranean range. Lat-lon shows the systematic equator-pole
# distortion notebook 02 already isolated.
#
# The next iterations of this notebook will add:
#
# - **rHEALPix** — equal-area projected variant of HEALPix (same family,
#   different cell shape)
# - **Mollweide** + **Behrmann** — equal-area cylindrical/pseudo-cylindrical
#   projections, gridded in projected space (atlas practice)
# - **EEA reference grid** — LAEA Europe, INSPIRE / Habitats Directive
#   standard for European biodiversity reporting
# - **ISEA3H** — the DGGS the Eco-ISEA3H paper advocates (requires the
#   Docker container at ghcr.io/annefou/dggs-biodiversity-bias:main with
#   DGGRID v8.41)
#
# Each addition produces another panel in the comparison figure.
