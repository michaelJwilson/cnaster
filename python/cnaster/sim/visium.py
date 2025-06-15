import os
import gzip
import logging
import polars
import numpy as np
from cnaster_rs import get_triangular_lattice

logger = logging.getLogger(__name__)

def generate_fake_barcodes(num_spots):
    return [f"VIS{i:05d}-1" for i in range(num_spots)]

def gen_visium(sample_dir, config, name):
    logger.info(f"Generating {name} visium.")

    # NB generate spot barcodes and positions
    nx, ny = config.visium.nx, config.visium.ny

    info = getattr(config.samples, name, None)

    height = info.height
    x0 = tuple(info.origin)
    
    lattice = get_triangular_lattice(nx, ny, height, x0=x0)
    barcodes = generate_fake_barcodes(nx * ny)

    tsv_path = f"{sample_dir}/{name}_visium.tsv.gz"

    with gzip.open(tsv_path, "wt") as f:
        f.write("# barcode\tx\ty\tz\n")
        
        for bc, (x, y, z) in zip(barcodes, lattice):
            f.write(f"{bc}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n")

    # NB generate umi counts
    #    - "exp_umi_per_spot": 3162,
    #    - "exp_snp_umi_per_spot": 501,
    
            
    logger.info(f"Generated visium to {sample_dir}")
