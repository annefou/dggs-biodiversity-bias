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
# # Part 4: What an ML kernel sees — feature-vector coherence for AI-ready biodiversity grids
#
# Modern biodiversity ML stacks heterogeneous layers — GBIF
# occurrences, ERA5 climate, Copernicus land cover, soil maps, MODIS
# vegetation indices — onto a **common grid** so every cell carries a
# feature vector the model can ingest as one observation. The model
# implicitly assumes that the *same cell index* refers to the *same
# geographic place* across every input layer. Whether that assumption
# holds depends on the grid.
#
# A convolutional model's receptive field is a **3×3 (or larger) cell
# window**. The model learns spatial operators that combine each
# cell's neighbours. For those operators to be transferable across a
# global dataset, the physical geography in a 3×3 kernel must look
# *the same shape* at every latitude. On rectilinear grids it doesn't.
#
# This notebook visualises exactly what a 3×3 ML kernel "sees" at
# **65°N, 15°E (boreal Scandinavia)** on three grid systems:
# **lat-lon 1°**, **Behrmann (cylindrical equal-area)**, and
# **HEALPix nside=64**. Same anchor point. Same kernel size. Different
# physical neighborhoods.

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

R = 6371.0  # Earth radius, km
DEG = np.pi / 180.0

ANCHOR_LAT = 65.0
ANCHOR_LON = 15.0
DISPLAY_HALF_EXTENT_KM = 1500e3  # ±1500 km LAEA window for display

# %% [markdown]
# ## Cell builders — one cell at a time, then a 3×3 window
#
# We reuse the same equal-area logic as Part 3 but at finer resolution
# (1° lat-lon, Behrmann tuned to the same equator width, HEALPix
# nside=64 with cell sides ~100 km). The 3×3 windows are constructed
# directly in each grid's native coordinates so the figure shows the
# kernel's true geographic footprint.

# %%
def latlon_cell_box(lat_c, lon_c, dlat, dlon, n_edge=20):
    lat_s, lat_n = lat_c - dlat / 2, lat_c + dlat / 2
    lon_w, lon_e = lon_c - dlon / 2, lon_c + dlon / 2
    bottom = np.linspace(lon_w, lon_e, n_edge)
    right = np.linspace(lat_s, lat_n, n_edge)
    top = np.linspace(lon_e, lon_w, n_edge)
    left = np.linspace(lat_n, lat_s, n_edge)
    lats = np.r_[np.full(n_edge, lat_s), right, np.full(n_edge, lat_n), left]
    lons = np.r_[bottom, np.full(n_edge, lon_e), top, np.full(n_edge, lon_w)]
    return lats, lons


def latlon_3x3_window(lat_c, lon_c, dlat=1.0, dlon=1.0):
    """9 lat-lon cells in a 3×3 grid centred on (lat_c, lon_c)."""
    cells = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            cells.append(latlon_cell_box(lat_c + di * dlat,
                                         lon_c + dj * dlon,
                                         dlat, dlon))
    return cells


def behrmann_dlat_at(lat_c, equator_dlat):
    """Latitude span of a Behrmann equal-area cell centred at lat_c
    whose equator counterpart spans `equator_dlat` degrees in lat."""
    sin_half = np.sin(equator_dlat / 2 * DEG)
    cos_c = np.cos(lat_c * DEG)
    arg = sin_half / cos_c
    arg = np.clip(arg, -0.9999, 0.9999)
    return 2 * np.degrees(np.arcsin(arg))


def behrmann_3x3_window(lat_c, lon_c, equator_dlat=1.0, equator_dlon=1.0):
    """9 Behrmann cells in a 3×3 grid centred on (lat_c, lon_c).

    Cells share Δx (longitude) and have lat spans matching their own
    centre latitude — so the window is built row by row from the
    central anchor.
    """
    # Central row: cell at lat_c with the right dlat.
    centre_dlat = behrmann_dlat_at(lat_c, equator_dlat)
    # The row above has its centre offset by (centre_dlat / 2 +
    # next_dlat / 2). We solve for the next-row centre by stepping in
    # latitude until the cumulative span matches.
    def step_centre(lat_anchor, direction):
        dlat_anchor = behrmann_dlat_at(lat_anchor, equator_dlat)
        # First-order estimate
        guess = lat_anchor + direction * dlat_anchor
        # Refine: row below/above has its own dlat
        dlat_next = behrmann_dlat_at(guess, equator_dlat)
        return lat_anchor + direction * (dlat_anchor + dlat_next) / 2

    lat_north = step_centre(lat_c, +1)
    lat_south = step_centre(lat_c, -1)

    cells = []
    for lat_row in (lat_south, lat_c, lat_north):
        dlat_row = behrmann_dlat_at(lat_row, equator_dlat)
        for dj in (-1, 0, 1):
            lon_cell = lon_c + dj * equator_dlon
            cells.append(latlon_cell_box(lat_row, lon_cell,
                                         dlat_row, equator_dlon))
    return cells


def healpix_3x3_window(lat_c, lon_c, nside):
    """HEALPix cell containing (lat_c, lon_c) plus its 8 ring neighbours.

    `hp.get_all_neighbours` returns the 8 cells around the central one
    in nested topology; combined with the centre, that is the 3×3
    analogue for HEALPix.
    """
    theta_c = (90.0 - lat_c) * DEG
    phi_c = (lon_c % 360) * DEG
    centre_pix = hp.ang2pix(nside, theta_c, phi_c, nest=True)
    neighbours = hp.get_all_neighbours(nside, theta_c, phi_c, nest=True)
    pixels = [centre_pix] + [int(p) for p in neighbours if p >= 0]
    cells = []
    for pix in pixels:
        xyz = hp.boundaries(nside, pix, step=8, nest=True)
        x, y, z = xyz
        lats = 90.0 - np.degrees(np.arccos(z))
        lons = np.degrees(np.arctan2(y, x))
        cells.append((lats, lons))
    return cells


# %% [markdown]
# ## Geographic-extent metrics
#
# For each window we report the kernel's true physical span: total
# north-south extent, total east-west extent, and the ratio (longer
# axis / shorter axis) — the closer to 1, the more "circular" and
# therefore consistent the receptive field is.

# %%
def laea_xy(lats, lons, lat0, lon0):
    lat_r = lats * DEG
    lon_r = lons * DEG
    lat0_r = lat0 * DEG
    lon0_r = lon0 * DEG
    cosc = (np.sin(lat0_r) * np.sin(lat_r)
            + np.cos(lat0_r) * np.cos(lat_r) * np.cos(lon_r - lon0_r))
    k = np.sqrt(2 / (1 + cosc))
    x = R * k * np.cos(lat_r) * np.sin(lon_r - lon0_r)
    y = R * k * (np.cos(lat0_r) * np.sin(lat_r)
                 - np.sin(lat0_r) * np.cos(lat_r) * np.cos(lon_r - lon0_r))
    return x, y


def window_metrics(cells, lat_c, lon_c):
    all_lats = np.concatenate([c[0] for c in cells])
    all_lons = np.concatenate([c[1] for c in cells])
    x, y = laea_xy(all_lats, all_lons, lat_c, lon_c)
    width = x.max() - x.min()
    height = y.max() - y.min()
    aspect = max(width, height) / min(width, height)
    return width, height, aspect


# %% [markdown]
# ## Build the figure — three panels, one anchor, one display projection
#
# All three panels share the **same Lambert azimuthal equal-area
# display map centred on (65°N, 15°E)**. Differences between panels
# come from the *grid system* alone, not from how we are looking at
# them.

# %%
NSIDE = 64
display_proj = ccrs.LambertAzimuthalEqualArea(
    central_latitude=ANCHOR_LAT, central_longitude=ANCHOR_LON
)

windows = [
    ("Lat-lon 1° × 1°", "tab:red",
     latlon_3x3_window(ANCHOR_LAT, ANCHOR_LON)),
    ("Behrmann (equal-area)", "tab:purple",
     behrmann_3x3_window(ANCHOR_LAT, ANCHOR_LON)),
    (f"HEALPix DGGS (nside={NSIDE})", "tab:blue",
     healpix_3x3_window(ANCHOR_LAT, ANCHOR_LON, NSIDE)),
]

fig = plt.figure(figsize=(15, 7.5))
gs = fig.add_gridspec(1, 3, wspace=0.05)

for col, (name, color, cells) in enumerate(windows):
    ax = fig.add_subplot(gs[0, col], projection=display_proj)
    ax.set_extent(
        [-DISPLAY_HALF_EXTENT_KM, DISPLAY_HALF_EXTENT_KM,
         -DISPLAY_HALF_EXTENT_KM, DISPLAY_HALF_EXTENT_KM],
        crs=display_proj,
    )
    ax.add_feature(cfeature.LAND, facecolor="0.92")
    ax.add_feature(cfeature.OCEAN, facecolor="0.96")
    ax.coastlines(linewidth=0.5, color="0.4")
    gl = ax.gridlines(linewidth=0.3, color="0.7", alpha=0.7)

    for lats, lons in cells:
        ax.fill(lons, lats, transform=ccrs.PlateCarree(),
                facecolor=color, alpha=0.40, edgecolor=color, linewidth=0.8)

    # Anchor point
    ax.plot(ANCHOR_LON, ANCHOR_LAT, marker="o", markersize=8,
            markerfacecolor="black", markeredgecolor="white",
            markeredgewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)

    width, height, aspect = window_metrics(cells, ANCHOR_LAT, ANCHOR_LON)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.text(
        0.02, 0.98,
        f"3×3 kernel at 65°N, 15°E\n"
        f"E–W span: {width:.0f} km\n"
        f"N–S span: {height:.0f} km\n"
        f"aspect:   {aspect:.1f}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="0.6",
                  boxstyle="round,pad=0.4", alpha=0.93),
    )

fig.suptitle(
    "What does a 3×3 ML kernel see at 65°N? "
    "— same anchor, same kernel size, different physical neighborhoods",
    fontsize=14, fontweight="bold", y=1.02,
)

# Subtitle / caption strip
fig.text(
    0.5, 0.02,
    "Models trained on a stacked feature cube (GBIF · ERA5 · Copernicus · soils · MODIS …) "
    "implicitly assume each cell index represents the same geographic place across all layers.\n"
    "Lat-lon and Behrmann break that assumption at high latitudes; HEALPix preserves it.",
    ha="center", va="bottom", fontsize=10, style="italic", color="0.25",
)

plt.savefig("../images/ai_kernel_footprint.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## What the figure shows
#
# All three panels are drawn on the same Lambert azimuthal equal-area
# map centred on **65°N, 15°E**. The black dot marks the anchor pixel.
# Every panel shows the **same conceptual operator** — a 3×3 ML kernel —
# but the *physical* geography that operator covers is wildly different:
#
# - **Lat-lon 1°.** A 3°×3° window. East-west extent shrinks with
#   latitude (1° lon at 65°N is 47 km, vs 111 km at the equator). The
#   kernel sees a thin, north-south oriented strip of geography. A
#   model learning to associate "ocean to the west, mountains to the
#   east" via such a kernel learns one operator near the equator and
#   a *different* one near the poles, even when the underlying biology
#   has the same structure.
#
# - **Behrmann (equal-area).** Equal area is preserved, so the
#   kernel's total km² is the same as at the equator. But the cells
#   become tall and narrow — E-W ground distance shrinks with
#   cos(φ) while N-S stretches by 1/cos(φ) to keep area constant.
#   At 65°N the 3×3 kernel sees a long vertical strip ~5× taller
#   than wide, spanning multiple biomes north-south while compressing
#   the east-west signal. Feature vectors at different latitudes
#   correspond to different physical neighbourhood shapes.
#
# - **HEALPix DGGS (nside=64).** Cells are compact at every latitude.
#   The 3×3 kernel sees a roughly circular ~250 km neighbourhood that
#   means the same geographic operator at the equator, in Scandinavia,
#   and in the Arctic. A model trained on a global stack is learning
#   one transferable operator.
#
# **The slide-5 takeaway.** The cost of a non-DGGS grid for AI work is
# not aesthetic; it is *feature-vector incoherence* across the
# training set. Equal-area cylindrical projections solve the area
# problem but leave the **shape** problem unsolved — and shape is
# what determines the geography a CNN's receptive field actually
# integrates over.
