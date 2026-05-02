# dggs-biodiversity-bias — proof that lat-lon grids systematically
# distort biodiversity counts, with a real-data demonstration on
# Quercus suber (GBIF) and a derivation of why DGGS (HEALPix) is
# the AI-ready answer.
#
# CPU-only image. Reproduces all six notebooks end-to-end via
# Snakemake. Dataset (~440 KB Quercus suber GBIF cache) is downloaded
# at first run by 02_gbif_quercus_suber.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        libgl1 \
        libglib2.0-0 \
        libgeos-dev \
        proj-bin \
        proj-data \
    && rm -rf /var/lib/apt/lists/*

# Build DGGRID from source for ISEA3H support via dggrid4py.
# Pure-Python ISEA3H libraries do not exist on Linux/macOS — vgrid's
# ISEA3H implementation is Windows-only (OpenEAGGR DLLs).
RUN git clone --depth 1 https://github.com/sahrk/DGGRID.git /opt/DGGRID \
    && cd /opt/DGGRID && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /opt/DGGRID

ENV DGGRID_PATH=/usr/local/bin/dggrid

RUN pip install --no-cache-dir \
        "numpy>=2.2,<2.3" \
        healpy \
        cartopy \
        matplotlib \
        pillow \
        requests \
        jupytext \
        nbclient \
        ipykernel \
        jupyter \
        snakemake \
        zenodo-get \
        h3 \
        dggrid4py \
        rhealpixdggs

WORKDIR /app
COPY . /app

# Default: reproduce all six notebooks end-to-end.
CMD ["snakemake", "--cores", "1", "all"]
