# dggs-biodiversity-bias

> Why equal-area DGGS cells matter for climate-driven biodiversity science — and why HEALPix is the right common DGGS for the *integration* of biodiversity with high-resolution Copernicus EO and Destination Earth climate models.

Supporting evidence for the EGU 2026 talk **"LifeWatch ERIC as Catalyst and Connector"** (EGU26-11348, ESSI2.6). Eight reproducible notebooks make a layered argument:

1. **Equal-area is necessary** *(notebooks 01–02)*. A regular lat-lon grid inflates biodiversity counts toward the equator by up to **23×** at 5° resolution. Mathematical property of the cells, not a sampling effect.
2. **Any of six equal-area choices passes the count test** *(notebook 07)*. HEALPix, H3, rHEALPix, ISEA3H, Mollweide, and the EEA reference grid (LAEA Europe / EPSG:3035 — INSPIRE / Habitats Directive standard) all agree on biodiversity density patterns over *Quercus suber*'s Mediterranean range. Choosing between them is not a count-correctness question.
3. **DGGS family preserves cell shape across latitudes; projection family does not** *(notebooks 03–04)*. Behrmann (aspect ≈ 5.0 at 65°N) and Mollweide distort poleward; EEA holds shape near its 52°N projection centre but distorts farther away. Every member of the DGGS family — HEALPix, H3, rHEALPix, ISEA3H — preserves compact cells at every latitude. **For ML pipelines stacking GBIF × Copernicus × ERA5 × MODIS into a single feature cube, this is what makes a CNN's receptive field mean the same geographic operator everywhere.**
4. **HEALPix is the right common DGGS for the *integration* future of biodiversity science** *(notebooks 06, 08)*. The case is not "HEALPix is uniquely best for biodiversity counts" — notebook 07 shows that any of six equal-area choices works. The case is that **biodiversity science is increasingly integrated with high-resolution Copernicus EO and Destination Earth climate models**, and on that integrated surface HEALPix has specific advantages the alternatives do not:
   - **Geometric deep learning on the sphere** is built on HEALPix (DeepSphere, spherical CNNs, equivariant networks). H3 / ISEA3H / rHEALPix have nothing comparable.
   - **Scattering networks for global EO data** — `foscat` (the FIESTA stack) operates on HEALPix natively.
   - **Sphere-harmonic transforms** (`healpy.map2alm` / `alm2map`) are native and fast on HEALPix; absent on the other DGGS.
   - **NESTED bit-shift hierarchical refinement** (parent = `pix >> 2`, children = `pix << 2 | k`) makes zoom-in / zoom-out **O(1) per cell** — no projection, no resampling, no hash lookup. Critical for tile-based Copernicus Zarr × biodiversity ML pipelines.
   - **Iso-latitude pixelization** makes zonal climate-zone analyses (latitudinal extinction risk, climate-band biodiversity stats) essentially free.
   - **A credible ellipsoidally-correct path** — via rHEALPix (already pip-installable) or "Ellipsoidal HEALPix" via the authalic-sphere mapping (the **ESA GRID4EARTH** approach) — addresses the systematic ~0.7% area bias at boreal latitudes that compounds in Copernicus×biodiversity stacks at high precision over decades.

   For **biodiversity-only counts at coarse resolution** any equal-area DGGS works; HEALPix is competitive but not uniquely necessary. **For the integrated future where biodiversity, climate models, and high-resolution EO data share one common DGGS**, HEALPix is the right substrate — not because it is "best for biodiversity" but because the climate-model and spherical-ML sides already live on it, and integration cost dominates.

## Headline numbers

**Step 1 — count bias of lat-lon (notebook 01–02):** at 5° resolution a 1 M-point uniform random distribution shows **23× more "occurrences per cell"** at the equator than at 85°N — purely from cell-area geometry, no biology.

**Step 3 — 3×3 ML-kernel aspect ratio at 65°N, 15°E (notebook 04):**

| Grid | Aspect ratio | Tier |
|---|---|---|
| Behrmann (equal-area projection) | **5.0** | Tier 2 (projection) |
| Mollweide 100 km | 1.3 | Tier 2 |
| Lat-lon 1° | 1.2 | Tier 1 (cautionary) |
| EEA reference grid 100 km | 1.0 | Tier 2 (azimuthal — preserves shape near 52°N centre) |
| rHEALPix res 4 | 1.2 | Tier 3 (DGGS) |
| HEALPix nside=64 | 1.3 | Tier 3 |
| H3 res 3 | 1.0 | Tier 3 |
| ISEA3H res 8 | 1.2 | Tier 3 |

**Step 4 — sphere-vs-WGS84 systematic area error (notebook 08):**

| Latitude | (HEALPix-on-sphere − WGS84) / WGS84 |
|---|---|
| 0°N | +0.45% |
| 30°N | +0.11% |
| 45°N | −0.22% |
| 65°N | **−0.65%** |
| 70°N | −0.73% |
| 85°N | −0.88% |

Total swing ~1.3 percentage points across populated latitudes — small per-cell, **systematic and compounding** across millions of 1 km cells × decades of climate-attribution data.

## Why this matters for climate-driven biodiversity science

The biodiversity work that informs conservation policy is not casual species-list aggregation. It is **climate-driven range-shift attribution, restoration outcome monitoring, Habitats Directive zonal reporting** — work where a 0.7% systematic bias compounds across cells × decades into real attribution errors. At that precision the choice of DGGS is a scientific-correctness issue, not an aesthetic one.

This repository connects to the **ESA GRID4EARTH** initiative ([Ellipsoidal HEALPix as a Common DGGS for Copernicus EO and Destination Earth](https://www.grid4earth.eu)) — the broader programme of building one ellipsoidally-correct, hierarchical, scalable DGGS for the European Earth-system data ecosystem. The argument made in this Jupyter Book is the biodiversity-side version of GRID4EARTH's case.

## Notebooks

All notebooks live in `notebooks/` as jupytext `.py` (committed) with `.ipynb` produced on demand by Snakemake.

| Step | Notebook | What it shows |
|---|---|---|
| 01 | `01_synthetic_proof.py` | Mathematical proof on 1 M uniform random points: lat-lon shows a fake equator-pole gradient, HEALPix is uniform. |
| 02 | `02_gbif_quercus_suber.py` | Real-data demonstration on **20,100 *Quercus suber* GBIF occurrences** — area variation across the species' Mediterranean range, lat-lon density vs HEALPix density. |
| 03 | `03_grid_anisotropy.py` | Cell-shape comparison at 0°, 40°N, 70°N for lat-lon, Behrmann, HEALPix. Aspect-ratio-vs-latitude curve. |
| 04 | `04_ai_kernel_footprint.py` | What an ML 3×3 kernel sees at 65°N, 15°E (Scandinavia) on three grids. The AI-readiness argument. |
| 05 | `05_equal_area_comparison.py` | 3-panel synthetic comparison: lat-lon vs Behrmann vs HEALPix on the same uniform data. Establishes that Behrmann and HEALPix both pass the count test. |
| 06 | `06_hierarchical_indexing.py` | One panel showing HEALPix NESTED refinement: parent cell at nside=8 → 16 children at nside=32 → 256 descendants at nside=128, exactly nested. |
| 07 | `07_multigrid_quercus_suber.py` | Comprehensive comparison of *Q. suber* density on **seven grids**: lat-lon (cautionary baseline), HEALPix nside=64 (DGGS, sphere), H3 res 3 (hexagonal DGGS), rHEALPix res 4 (DGGS, WGS84 ellipsoid), Mollweide ~100 km (equal-area projection), EEA reference grid 100 km (LAEA Europe / EPSG:3035, INSPIRE / Habitats Directive standard), and ISEA3H res 8 (the system the Eco-ISEA3H paper advocates). Confirms that **all six equal-area choices agree** on the apparent density pattern; lat-lon is the only one that distorts. |
| 08 | `08_sphere_vs_ellipsoid.py` | **HEALPix-specific advantages and refinements.** Section A: HEALPix-on-sphere vs HEALPix-on-WGS84 systematic area error (~0.7% at boreal latitudes); rHEALPix and "Ellipsoidal HEALPix" via authalic-sphere (GRID4EARTH) as the two solutions. Section B: NESTED bit-shift hierarchical refinement — `parent = pix >> 2`, `children = pix << 2 \| k`, verified against `healpy.pix2ang`/`ang2pix`. Section C: iso-latitude pixelization — every HEALPix ring sits at exactly one latitude, making zonal climate-biodiversity analyses essentially free; visual contrast against H3 hex tessellation. |

## How to reproduce

End-to-end via Snakemake (recommended):

```bash
mamba env create -f environment.yml
mamba activate dggs-biodiversity-bias
snakemake --cores 1 all
```

This converts every `.py` notebook to `.ipynb`, executes it, and writes figures to `images/`. The Quercus suber GBIF cache (`data/quercus_suber_gbif.json`, ~440 KB, 20,100 records) is downloaded on first run by notebook 02 — about 15–20 minutes due to GBIF search-API throttling on deep offsets. Subsequent runs read the cache.

Single-notebook run:

```bash
jupytext --to notebook notebooks/01_synthetic_proof.py
jupyter execute --inplace notebooks/01_synthetic_proof.ipynb
```

Container:

```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/images:/app/images \
  ghcr.io/annefou/dggs-biodiversity-bias:latest
```

## Data provenance

- **Quercus suber occurrences** — GBIF Occurrence Search API, taxon key 2879411, filter `hasCoordinate=true&hasGeospatialIssue=false`. Downloaded by notebook 02 at first run; cache file lives in `data/`.
- **All other data is synthetic** — generated from a uniform-random point distribution on the sphere within the notebooks themselves.

## How to cite

This repository is archived on Zenodo (DOI minted on first GitHub release). See `CITATION.cff` and `codemeta.json` for the full citation graph.

Key foundational references the work depends on:

- **Górski et al. 2005** — HEALPix design and equal-area property. *ApJ* 622:759. [doi:10.1086/427976](https://doi.org/10.1086/427976)
- **Sahr et al. 2003** — DGGS as an equal-area-by-design framework. *CaGIS* 30(2):121–134. [doi:10.1559/152304003100011090](https://doi.org/10.1559/152304003100011090)
- **Hauffe et al. 2023** — Equal-area DGGS (ISEA3H) for biodiversity ML, ~900 mammal species. *Sci Data* 10:77. [doi:10.1038/s41597-023-01966-x](https://doi.org/10.1038/s41597-023-01966-x)
- **Kmoch et al. 2022** — Area and shape distortions in open-source DGGS implementations. *Big Earth Data* 6(3):256–275. [doi:10.1080/20964471.2022.2094926](https://doi.org/10.1080/20964471.2022.2094926)

## Credits

Anne Fouilloux — LifeWatch ERIC ([orcid:0000-0002-1784-2920](https://orcid.org/0000-0002-1784-2920)).

Part of the **Science Live** platform ([sciencelive4all.org](https://platform.sciencelive4all.org)) and the **FAIR2Adapt** EOSC project.

## License

Code: **MIT** (see `LICENSE`).
Generated figures and notebook prose: **CC-BY 4.0**.
