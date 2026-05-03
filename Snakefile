NOTEBOOKS = [
    "01_synthetic_proof",
    "02_gbif_quercus_suber",
    "03_grid_anisotropy",
    "04_ai_kernel_footprint",
    "05_equal_area_comparison",
    "06_hierarchical_indexing",
    "07_multigrid_quercus_suber",
]

IMAGES = {
    "01_synthetic_proof":         ["images/map_raw_counts.png",
                                   "images/raw_counts_by_latitude.png",
                                   "images/density_vs_latitude.png",
                                   "images/density_histogram.png"],
    "02_gbif_quercus_suber":      ["images/gbif_quercus_suber.png"],
    "03_grid_anisotropy":         ["images/grid_anisotropy.png"],
    "04_ai_kernel_footprint":     ["images/ai_kernel_footprint.png"],
    "05_equal_area_comparison":   ["images/equal_area_comparison.png"],
    "06_hierarchical_indexing":   ["images/hierarchical_indexing.png"],
    "07_multigrid_quercus_suber": ["images/multigrid_quercus_suber.png"],
}


rule all:
    input:
        [img for nb in NOTEBOOKS for img in IMAGES[nb]],


rule run_notebook:
    input:
        script="notebooks/{name}.py",
    output:
        notebook="notebooks/{name}.ipynb",
    shell:
        """
        jupytext --to notebook {input.script}
        jupyter execute --inplace {output.notebook}
        """


# One concrete target per notebook so Snakemake can resolve outputs.
for _nb, _imgs in IMAGES.items():
    rule:
        name:
            f"images_{_nb}"
        input:
            f"notebooks/{_nb}.ipynb",
        output:
            *_imgs
        shell:
            "true  # outputs are produced by run_notebook"
