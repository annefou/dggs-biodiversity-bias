# Equal-area cells matter for biodiversity

Supporting evidence for the EGU 2026 talk **"LifeWatch ERIC as Catalyst and Connector"** (EGU26-11348, ESSI2.6, 3–8 May 2026, Vienna).

## The argument in one paragraph

Global biodiversity analyses must combine data from many sources onto a common grid before any model — statistical or machine-learned — can use them. The grid we choose is not a visualisation choice; it is a statistical one. A regular latitude-longitude grid systematically over-counts biodiversity at low latitudes and under-counts it at high latitudes, by up to a factor of 23 at 5° resolution, purely because grid cells shrink poleward. **Equal-area** projections (Behrmann, Mollweide) — the recognised standard for global biodiversity maps — solve that count bias. But for AI-ready, multi-resolution, cloud-native pipelines, equal-area is necessary and not sufficient: at high latitudes a Behrmann cell becomes a tall thin strip, and a 3×3 ML kernel covers a wildly different physical neighbourhood than at the equator. The same cell index stops meaning the same place across stacked feature layers — a quiet failure mode in models trained on heterogeneous biodiversity stacks. Discrete Global Grid Systems (HEALPix in particular) preserve equal area *and* compact cell shape *and* a deterministic hierarchical refinement.

## What this Jupyter Book contains

Seven notebooks, each one a single, focused piece of evidence:

1. **{doc}`Synthetic proof <notebooks/01_synthetic_proof>`** — 1,000,000 uniform random points on the sphere binned on a 5° lat-lon grid versus HEALPix. The lat-lon panel develops a fake equator-pole gradient; HEALPix is uniform. The artefact is mathematical, not statistical.

2. **{doc}`Real biodiversity data <notebooks/02_gbif_quercus_suber>`** — 20,100 *Quercus suber* (cork oak) occurrences from GBIF, binned on a 1° lat-lon grid and on HEALPix nside=64. Within the species' Mediterranean range, lat-lon cell areas vary by ~23%; the same true density therefore appears as 23% more "occurrences per cell" in southern cells than northern, with no ecological content.

3. **{doc}`Cell-shape anisotropy <notebooks/03_grid_anisotropy>`** — for 1° lat-lon, 5° Behrmann (cylindrical equal-area), and HEALPix nside=16, render a representative cell at 0°, 40°N, 70°N and report aspect ratio. HEALPix stays close to 1; lat-lon and Behrmann diverge.

4. **{doc}`What an ML kernel sees <notebooks/04_ai_kernel_footprint>`** — at 65°N, 15°E (boreal Scandinavia), draw the 3×3 kernel of an ML model on each grid. Behrmann's kernel covers a 159 × 796 km vertical strip (aspect 5.0); HEALPix's covers a compact 502 × 562 km neighbourhood (aspect 1.3).

5. **{doc}`Three-grid comparison <notebooks/05_equal_area_comparison>`** — lat-lon vs Behrmann vs HEALPix on the same 1 M uniform points. Behrmann and HEALPix are both uniform; the count bias is a solved problem under any equal-area scheme.

6. **{doc}`Hierarchical indexing <notebooks/06_hierarchical_indexing>`** — a single HEALPix NESTED parent cell (nside=8), exactly tiled by 16 children (nside=32) and 256 descendants (nside=128), drawn over Scandinavia. Refinement is a deterministic bit-shift on the cell index — no projection, no interpolation, no resampling.

7. **{doc}`Comprehensive multi-grid comparison <notebooks/07_multigrid_quercus_suber>`** — same Q. suber GBIF data on **seven grids**: lat-lon (cautionary), HEALPix nside=64, H3 res 3, rHEALPix res 4, Mollweide ~100 km, the **EEA reference grid** (LAEA Europe / EPSG:3035 — the INSPIRE / Habitats Directive standard for European biodiversity reporting), and **ISEA3H res 8** (the system the Eco-ISEA3H paper advocates). All six equal-area choices agree on the apparent density pattern; lat-lon is the only one that distorts. Establishes that equal-area is the load-bearing property — what notebooks 03–06 then add is the case for HEALPix specifically (compact shape *across all latitudes* and AI-ready hierarchical refinement, properties that distinguish DGGS from projection-based equal-area grids).

## How to use this material

The notebooks are designed to be read in order, but each one is self-contained: you can drop into any single notebook and it will run on its own. Notebook 02 downloads ~440 KB of GBIF data on first run (15–20 minutes due to GBIF API throttling); subsequent runs read the cache. All other notebooks are pure synthetic.

To run everything end-to-end, see the README — `snakemake --cores 1 all` reproduces every figure.

## Citation

If this material is useful in your own work please cite the repository (DOI on first release) and the foundational references in `CITATION.cff` — Górski et al. 2005 (HEALPix), Sahr et al. 2003 (DGGS), Hauffe et al. 2023 (DGGS for biodiversity), Kmoch et al. 2022 (DGGS area distortions).
