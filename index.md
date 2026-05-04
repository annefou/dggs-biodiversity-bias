# A common DGGS for the integration of biodiversity, climate, and EO

Supporting evidence for the EGU 2026 talk **"LifeWatch ERIC as Catalyst and Connector"** (EGU26-11348, ESSI2.6, 3–8 May 2026, Vienna).

## The argument, in four steps

Global biodiversity analyses must combine GBIF occurrences, ERA5 climate, Copernicus land cover, and Destination Earth model output onto a **single common grid** before any statistical or machine-learned model can use them. The grid choice is not a visualisation question — it is a scientific-correctness question, especially for **climate-driven range-shift attribution, restoration outcome monitoring, and Habitats Directive zonal reporting**, where small systematic biases compound across millions of cells × decades into real attribution errors.

The argument the eight notebooks make:

1. **Equal-area is necessary.** Regular latitude-longitude grids systematically over-count biodiversity at low latitudes and under-count it at high latitudes, by up to a factor of **23×** at 5° resolution — purely because lat-lon cells shrink poleward. Mathematical property of the cells, not a sampling effect (notebooks 01–02).
2. **Any of six equal-area choices passes the count test.** Notebook 07 compares HEALPix, H3, rHEALPix, ISEA3H, Mollweide, and the EEA reference grid (LAEA Europe / EPSG:3035) on the same 20,100-record GBIF *Quercus suber* dataset. All six agree on the apparent density pattern. **For biodiversity counts in isolation, the choice between them is not a count-correctness question.**
3. **DGGS family preserves cell shape across latitudes; projection family does not.** Notebooks 03–04 show that Behrmann (cylindrical equal-area) at 65°N produces 3×3 ML kernels with **aspect ratio 5.0** — vertical strips that span multiple biomes north-south. Mollweide and EEA distort more gently but distort. The DGGS family — HEALPix, H3, rHEALPix, ISEA3H — all preserve compact (~aspect-1) cells everywhere. For ML pipelines stacking features into a single cube, this is what makes a CNN's receptive field mean the same geographic operator everywhere on Earth.
4. **HEALPix is the right common DGGS for the *integration* of biodiversity with Copernicus EO and Destination Earth climate models** *(notebooks 06, 08)*. The case is **not** that HEALPix is uniquely best for biodiversity counts — step 2 establishes that any of the six equal-area choices works. The case is that **biodiversity science is increasingly integrated with high-resolution EO and climate models**, and on that integrated surface HEALPix has specific advantages the alternatives do not:
   - **Geometric deep learning on the sphere** (DeepSphere, spherical CNNs, equivariant networks) is built on HEALPix; H3 / ISEA3H / rHEALPix have nothing comparable.
   - **Scattering networks** for global EO and biodiversity feature extraction — `foscat` (the FIESTA stack) operates on HEALPix natively.
   - **Sphere-harmonic transforms** (`healpy.map2alm` / `alm2map`) are native to HEALPix; absent on the other DGGS.
   - **NESTED bit-shift hierarchical refinement** (`parent = pix >> 2`, `children = pix << 2 | k`) makes zoom-in / zoom-out *O(1)* per cell — critical for tile-based Copernicus Zarr × biodiversity ML pipelines.
   - **Iso-latitude pixelization** makes zonal climate-zone analyses essentially free.
   - **A credible ellipsoidally-correct path** — via rHEALPix (already pip-installable) or "Ellipsoidal HEALPix" via the authalic-sphere mapping (the **ESA GRID4EARTH** approach) — addresses the systematic ~0.7% area bias at boreal latitudes that HEALPix-on-sphere otherwise compounds across decades of integrated Copernicus × biodiversity data.

   The honest conclusion: for **biodiversity counts at coarse resolution in isolation**, any equal-area DGGS works and the choice is largely about ecosystem (EEA grid is the European regulatory standard; H3 has GIS-industry momentum; ISEA3H has Eco-ISEA3H paper alignment). For **biodiversity integrated with Copernicus EO and climate models on a single common DGGS** — the GRID4EARTH future — HEALPix is the right substrate, not because it is "best for biodiversity" but because the climate-model and spherical-ML sides already live on it, and integration cost dominates.

## What this Jupyter Book contains

Eight notebooks, each one a single, focused piece of evidence:

1. **{doc}`Synthetic proof <notebooks/01_synthetic_proof>`** — 1,000,000 uniform random points on the sphere binned on a 5° lat-lon grid versus HEALPix. The lat-lon panel develops a fake equator-pole gradient; HEALPix is uniform. The artefact is mathematical, not statistical.

2. **{doc}`Real biodiversity data <notebooks/02_gbif_quercus_suber>`** — 20,100 *Quercus suber* (cork oak) occurrences from GBIF, binned on a 1° lat-lon grid and on HEALPix nside=64. Within the species' Mediterranean range, lat-lon cell areas vary by ~23%; the same true density therefore appears as 23% more "occurrences per cell" in southern cells than northern, with no ecological content.

3. **{doc}`Cell-shape anisotropy <notebooks/03_grid_anisotropy>`** — for 1° lat-lon, 5° Behrmann (cylindrical equal-area), and HEALPix nside=16, render a representative cell at 0°, 40°N, 70°N and report aspect ratio. HEALPix stays close to 1; lat-lon and Behrmann diverge.

4. **{doc}`What an ML kernel sees <notebooks/04_ai_kernel_footprint>`** — at 65°N, 15°E (boreal Scandinavia), draw the 3×3 kernel of an ML model on each grid. Behrmann's kernel covers a 159 × 796 km vertical strip (aspect 5.0); HEALPix's covers a compact 502 × 562 km neighbourhood (aspect 1.3).

5. **{doc}`Three-grid comparison <notebooks/05_equal_area_comparison>`** — lat-lon vs Behrmann vs HEALPix on the same 1 M uniform points. Behrmann and HEALPix are both uniform; the count bias is a solved problem under any equal-area scheme.

6. **{doc}`Hierarchical indexing <notebooks/06_hierarchical_indexing>`** — a single HEALPix NESTED parent cell (nside=8), exactly tiled by 16 children (nside=32) and 256 descendants (nside=128), drawn over Scandinavia. Refinement is a deterministic bit-shift on the cell index — no projection, no interpolation, no resampling.

7. **{doc}`Comprehensive multi-grid comparison <notebooks/07_multigrid_quercus_suber>`** — same *Quercus suber* GBIF data on **eight grids**: lat-lon (cautionary), HEALPix nside=64 (sphere), **HEALPix-geo nside=64** (WGS84 via authalic-sphere mapping — the **GRID4EARTH** path), H3 res 3, rHEALPix res 4 (WGS84 cube-projected), Mollweide ~100 km, the **EEA reference grid** (LAEA Europe / EPSG:3035 — the INSPIRE / Habitats Directive standard), and **ISEA3H res 8** (the system the Eco-ISEA3H paper advocates). All seven equal-area choices agree on the apparent density pattern; lat-lon is the only one that distorts. Establishes that equal-area is the load-bearing property — what notebooks 03–06 and 08 then add is the case for HEALPix specifically.

8. **{doc}`HEALPix-specific advantages and refinements <notebooks/08_sphere_vs_ellipsoid>`** — three HEALPix-family properties that make it the right common DGGS for climate-driven biodiversity science:
   - **Section A — Sphere vs WGS84 ellipsoid.** HEALPix is on the sphere; biodiversity occurrences and Copernicus EO products are on WGS84. The mismatch is ~0.7% systematic area error at boreal latitudes — small per-cell, **systematic and compounding** across millions of 1 km cells × decades of climate-attribution data. Two solutions: rHEALPix (already used in notebook 07) and "Ellipsoidal HEALPix" via authalic-sphere mapping (the **ESA GRID4EARTH** approach).
   - **Section B — NESTED bit-shift refinement.** Parent and children of any HEALPix cell are pure integer arithmetic (`parent = pix >> 2`, `children = pix << 2 | k`). Zoom in / zoom out is **O(1) per cell**, no projection, no resampling — critical for Copernicus Zarr × biodiversity tile pipelines.
   - **Section C — Iso-latitude pixelization.** Every HEALPix cell sits on a fixed-colatitude ring; zonal climate-zone analyses are essentially free. H3 hex tessellation breaks this property.

## How to use this material

The notebooks are designed to be read in order, but each one is self-contained: you can drop into any single notebook and it will run on its own. Notebook 02 downloads ~440 KB of GBIF data on first run (15–20 minutes due to GBIF API throttling); subsequent runs read the cache. All other notebooks are pure synthetic.

To run everything end-to-end, see the README — `snakemake --cores 1 all` reproduces every figure.

## Connection to ESA GRID4EARTH

This Jupyter Book is the biodiversity-side version of the case the **ESA GRID4EARTH** initiative makes for **Ellipsoidal HEALPix as a Common DGGS** ([grid4earth.eu](https://www.grid4earth.eu)) — bridging spherical climate models (Destination Earth) and ellipsoidal Earth-observation data (Copernicus) on a single ellipsoidally-correct, hierarchical, scalable DGGS. Notebook 08 Section A makes the biodiversity-precision argument for that bridge.

## Citation

If this material is useful in your own work please cite the repository (DOI on first release) and the foundational references in `CITATION.cff` — Górski et al. 2005 (HEALPix), Sahr et al. 2003 (DGGS), Hauffe et al. 2023 (DGGS for biodiversity), Kmoch et al. 2022 (DGGS area distortions).
