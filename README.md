# dggs-biodiversity-bias

> Why equal-area cells matter for biodiversity — and why DGGS adds shape and hierarchy that equal-area projections do not.

Supporting evidence for the EGU 2026 talk **"LifeWatch ERIC as Catalyst and Connector"** (EGU26-11348, ESSI2.6). Seven reproducible notebooks make a single argument:

- **Equal-area is necessary.** A regular lat-lon grid inflates biodiversity counts toward the equator by up to **23×** at 5° resolution, purely from grid geometry. This is a mathematical property of the cells, not a sampling effect.
- **Equal-area is not sufficient for AI-ready data.** Behrmann and Mollweide projections fix the count bias but at high latitudes their cells become tall thin strips (aspect ratio ≈ 5 at 65°N for a 1° equator cell). A 3×3 ML kernel covers wildly different physical neighbourhoods at different latitudes; feature vectors stop meaning the same thing across a stacked feature cube.
- **HEALPix preserves both.** Equal-area cells *and* compact, hierarchically refinable shape *and* a NESTED quadtree where refinement is a deterministic bit-shift on the cell index — no projection, no resampling.

## Headline numbers

| Resolution | Lat-lon equator (5°) | Lat-lon at 85° | Behrmann at 65°N (3×3) | HEALPix at 65°N (3×3) |
|---|---|---|---|---|
| Cell area | ~615 ·10³ km² | ~27 ·10³ km² | constant | constant |
| Mean count per cell (1 M points) | ~606 | ~26 (**23× bias**) | uniform | uniform |
| 3×3 kernel aspect ratio | 2.2 | n/a | **5.0** | **1.3** |

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
