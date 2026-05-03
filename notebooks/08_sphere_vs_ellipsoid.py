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
# # Part 8: HEALPix-specific advantages and refinements
#
# Notebook 07 established that **all six equal-area aggregation
# choices** (HEALPix, H3, rHEALPix, Mollweide, EEA reference grid,
# ISEA3H) agree on biodiversity density patterns; lat-lon is the only
# one that distorts. Notebooks 03 and 04 established that the **DGGS
# family** (HEALPix, H3, rHEALPix, ISEA3H) preserves compact cell
# shape across latitudes, while the projection family (Behrmann,
# Mollweide, partly EEA) does not.
#
# So why HEALPix specifically? This notebook isolates three
# **HEALPix-family advantages** that distinguish it from the other
# DGGS choices and matter for climate-driven biodiversity science:
#
# - **Section A — Sphere vs WGS84 ellipsoid.** HEALPix is defined on
#   the unit sphere. EO data, GBIF occurrences, and Copernicus
#   products are referenced to the WGS84 ellipsoid. The mismatch is
#   small (~0.7% area error at boreal latitudes) but **systematic and
#   compounding** — exactly the regime where climate-driven biodiversity
#   attribution conclusions live or die. Solutions: **rHEALPix**
#   (HEALPix on WGS84 directly) and **Ellipsoidal HEALPix** via the
#   authalic-sphere mapping (the GRID4EARTH approach).
# - **Section B — NESTED bit-shift refinement.** Parent and children
#   of any HEALPix cell are computed by integer bit operations alone
#   (parent = `pix >> 2`, children = `pix << 2 | k`). This makes
#   zoom-in / zoom-out **O(1) per cell**, with no projection, no
#   resampling, no hash lookup.
# - **Section C — Iso-latitude pixelization.** All HEALPix cells in
#   the same "ring" share the same colatitude. This makes
#   latitude-banded analyses (zonal extinction risk, latitudinal
#   phenology, climate-zone aggregation) trivially fast.

# %% [markdown]
# ## Section A — HEALPix on the sphere vs HEALPix on the WGS84 ellipsoid
#
# The Earth is not a sphere. WGS84 — the geodetic datum every
# Copernicus product, every GBIF occurrence, every Destination Earth
# climate model output uses — is an oblate ellipsoid with flattening
# ~1/298.26. HEALPix's mathematics, by contrast, is defined on the
# **unit sphere** (Górski et al. 2005).
#
# When we feed a (lat, lon) pair into `hp.ang2pix(nside, theta, phi)`,
# we are asking "which HEALPix cell does this point fall into,
# treating it as if it were on a sphere?" — and the cell areas HEALPix
# uses are spherical cell areas, not the area of the equivalent patch
# on the WGS84 ellipsoid where the data actually lives.
#
# The error this introduces is **small in percentage terms** but
# **systematic in latitude** and **compounding across cells and
# decades** — exactly the regime where climate-biodiversity attribution
# conclusions live or die. A 0.7% area bias at 65°N, multiplied by
# millions of 1 km cells × decades of stacked Copernicus + biodiversity
# data, is a real source of systematic bias in zonal density estimates,
# range-shift attribution, and habitat-degradation reporting.
#
# This section quantifies the mismatch and shows the two existing
# paths that solve it.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Geod

WGS84 = Geod(ellps="WGS84")

# A few canonical sphere-radius choices used in HEALPix-on-sphere work:
R_MEAN_KM = 6371.0          # mean Earth radius (the "obvious" choice)
R_AUTHALIC_KM = 6371.0072   # authalic sphere — same total surface area as WGS84
R_VOLUME_KM = 6371.0008     # volume-equivalent sphere
A_WGS84_KM = 6378.137       # WGS84 semi-major axis
B_WGS84_KM = 6356.752314245 # WGS84 semi-minor axis (derived from f = 1/298.257223563)

# %% [markdown]
# ### Cell area: sphere vs WGS84 ellipsoid
#
# We pick a 1° × 1° geodetic-coordinates cell at latitude φ (the
# typical biodiversity-aggregation resolution at country scale) and
# compute its area:
#
# - on the unit sphere (mean / authalic radius), treating geodetic lat
#   as if it were spherical colatitude;
# - on the WGS84 ellipsoid, using `pyproj.Geod.polygon_area_perimeter`
#   which solves the geodesic exactly.
#
# The discrepancy is what HEALPix-on-sphere systematically gets wrong.

# %%
def cell_area_sphere(lat_c, dlat=1.0, dlon=1.0, radius_km=R_MEAN_KM):
    """Area (km²) of a geodetic 1°×1° cell, treated as spherical."""
    lat_n = np.radians(lat_c + dlat / 2)
    lat_s = np.radians(lat_c - dlat / 2)
    return radius_km ** 2 * np.radians(dlon) * (np.sin(lat_n) - np.sin(lat_s))


def cell_area_wgs84(lat_c, dlat=1.0, dlon=1.0):
    """True WGS84-ellipsoidal area (km²) of a 1°×1° cell."""
    lons = [-dlon / 2, dlon / 2, dlon / 2, -dlon / 2]
    lats = [lat_c - dlat / 2, lat_c - dlat / 2,
            lat_c + dlat / 2, lat_c + dlat / 2]
    area_m2, _ = WGS84.polygon_area_perimeter(lons, lats)
    return abs(area_m2) / 1e6  # km²


lat_grid = np.linspace(0, 89, 90)
A_mean = np.array([cell_area_sphere(l, radius_km=R_MEAN_KM) for l in lat_grid])
A_authalic = np.array([cell_area_sphere(l, radius_km=R_AUTHALIC_KM) for l in lat_grid])
A_wgs84 = np.array([cell_area_wgs84(l) for l in lat_grid])

err_mean_pct = 100 * (A_mean - A_wgs84) / A_wgs84
err_authalic_pct = 100 * (A_authalic - A_wgs84) / A_wgs84

print("Sample cell-area errors (1°×1° cell, sphere vs WGS84):")
print(f"  {'lat':>5} {'A_sphere(km²)':>14} {'A_wgs84(km²)':>14} {'err_mean(%)':>12} {'err_authalic(%)':>16}")
for lat in [0, 30, 45, 65, 70, 85]:
    i = int(lat)
    print(f"  {lat:>5} {A_mean[i]:>14.1f} {A_wgs84[i]:>14.1f} "
          f"{err_mean_pct[i]:>+12.4f} {err_authalic_pct[i]:>+16.4f}")

# %% [markdown]
# ### Why this matters for climate-driven biodiversity science
#
# The maximum percentage error is small (~0.45% at the equator,
# ~−0.88% at 85°N). At first glance this looks negligible compared to
# the count bias of lat-lon (up to 23× at 5° resolution; notebook 02).
#
# But the **structure** of the error is what makes it a problem:
#
# 1. **Systematic, not random.** The error always has the same sign at
#    a given latitude. It does not average out.
# 2. **Latitude-dependent.** The error swings ~1.3 percentage points
#    across the populated latitude range. Zonal comparisons (boreal
#    vs temperate vs tropical) accumulate this swing as a bias.
# 3. **Compounds across cells and decades.** A climate-impact
#    attribution analysis that combines Copernicus EO products at
#    1 km × decades of data stack contains O(10^9) cell-years of
#    estimates. A 0.7% systematic bias on each is the difference
#    between "detecting" and "missing" a climate-driven biodiversity
#    signal at the threshold of statistical significance.
# 4. **Compounds across products.** If GBIF occurrences are aggregated
#    on HEALPix-on-sphere, ERA5 climate is aggregated on its native
#    Gaussian grid then resampled, and Copernicus land cover is on
#    yet another reference, the cross-product comparison inherits the
#    union of all the per-product systematic errors.
#
# This is precisely the regime that **GRID4EARTH (ESA)** identifies as
# requiring a common, ellipsoidally-correct DGGS for the
# Copernicus × Destination Earth integration.

# %% [markdown]
# ### The figure — sphere-vs-WGS84 area error vs latitude

# %%
fig = plt.figure(figsize=(13, 5.5))
gs = fig.add_gridspec(1, 2, wspace=0.30)

# -- Left: cell areas in km² ----------------------------------------
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(lat_grid, A_mean, color="tab:red", lw=2,
          label=f"Sphere R={R_MEAN_KM:.0f} km (mean radius)")
ax_a.plot(lat_grid, A_authalic, color="tab:orange", lw=1.4, linestyle="--",
          label=f"Sphere R={R_AUTHALIC_KM:.4f} km (authalic — equal total area)")
ax_a.plot(lat_grid, A_wgs84, color="tab:blue", lw=2,
          label="WGS84 ellipsoid (pyproj.Geod, exact)")
ax_a.set_xlabel("Latitude (°N)")
ax_a.set_ylabel("Cell area (km²) of a 1° × 1° geodetic cell")
ax_a.set_title("Area of a 1° × 1° cell — sphere vs WGS84 ellipsoid",
               fontsize=11, fontweight="bold")
ax_a.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax_a.set_xlim(0, 89)
ax_a.grid(alpha=0.3)

# -- Right: % error vs latitude -------------------------------------
ax_b = fig.add_subplot(gs[0, 1])
ax_b.fill_between(lat_grid, 0, err_mean_pct,
                  where=err_mean_pct >= 0, color="tab:red", alpha=0.18,
                  label="Sphere-on-WGS84 over-estimates area")
ax_b.fill_between(lat_grid, 0, err_mean_pct,
                  where=err_mean_pct < 0, color="tab:blue", alpha=0.18,
                  label="Sphere-on-WGS84 under-estimates area")
ax_b.plot(lat_grid, err_mean_pct, color="tab:red", lw=2,
          label=f"Sphere R={R_MEAN_KM:.0f} km (mean)")
ax_b.plot(lat_grid, err_authalic_pct, color="tab:orange", lw=1.4,
          linestyle="--",
          label=f"Sphere R={R_AUTHALIC_KM:.2f} km (authalic)")
ax_b.axhline(0, color="0.4", linewidth=0.7)
ax_b.set_xlabel("Latitude (°N)")
ax_b.set_ylabel("(A_sphere − A_WGS84) / A_WGS84   (%)")
ax_b.set_title(
    "Systematic area error of HEALPix-on-sphere\n"
    "across the populated latitude range",
    fontsize=11, fontweight="bold",
)
ax_b.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
ax_b.set_xlim(0, 89)
ax_b.grid(alpha=0.3)

# Annotation box at boreal latitudes
ax_b.annotate(
    f"At 65°N: {err_mean_pct[65]:+.3f}%\n"
    f"At 70°N: {err_mean_pct[70]:+.3f}%\n"
    f"At 85°N: {err_mean_pct[85]:+.3f}%",
    xy=(70, err_mean_pct[70]), xytext=(45, -0.4),
    fontsize=8.5, family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="0.6", alpha=0.95),
    arrowprops=dict(arrowstyle="->", color="0.4"),
)

fig.suptitle(
    "HEALPix-on-sphere vs HEALPix-on-WGS84 — small per-cell error, "
    "systematic across latitude, compounding for climate-biodiversity "
    "attribution",
    fontsize=12, fontweight="bold", y=1.02,
)

plt.savefig("../images/sphere_vs_ellipsoid_area_error.png",
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Two existing solutions
#
# **rHEALPix (already in this study).** rHEALPix defines HEALPix-style
# equal-area cells **directly on the WGS84 ellipsoid**, via a cube
# projection. Cells have exactly equal area on the actual Earth shape
# — no mean / authalic / volume sphere choice needed. We used
# `rhealpixdggs` (PyPI) in notebook 07; it is the pip-installable,
# production-ready answer for an ellipsoid-native HEALPix family
# member.
#
# **Ellipsoidal HEALPix via authalic-sphere mapping (GRID4EARTH).**
# The GRID4EARTH (ESA) approach is to map the WGS84 ellipsoid to its
# **authalic sphere** (the sphere with the same total surface area)
# via the standard authalic-latitude transform. HEALPix defined on
# the authalic sphere then yields cells of exactly equal area on the
# *original* WGS84 ellipsoid — preserving the equal-area property
# while using the well-established HEALPix code path. This bridges
# spherical climate models (Destination Earth) and ellipsoidal
# Earth-observation data (Copernicus) on a single common DGGS.
#
# Both approaches eliminate the per-latitude ~0.7% systematic bias
# documented in this section. Choosing between them is a matter of
# code-base alignment: `rhealpixdggs` is mature and pip-installable;
# Ellipsoidal HEALPix lives inside the GRID4EARTH ecosystem and is
# the path that integrates with the Destination Earth Common Data
# Model.
#
# **Section B (NESTED bit-shift) and Section C (iso-latitude) — to
# follow.** Both are pure HEALPix-family properties that survive the
# sphere → authalic mapping unchanged.
