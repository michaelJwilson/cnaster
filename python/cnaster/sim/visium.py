import os
import gzip
import glob
import logging
import polars
import numpy as np
from cnaster.sim.clone import Clone
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

    # TODO HARDCODE
    clones = [
        Clone(xx)
        for xx in sorted(
            glob.glob(config.output_dir + f"/phylogenies/phylogeny2/*.json")
        )
    ]

    num_segments = config.mappable_genome_kbp // config.segment_size_kbp
    
    for bc, (x, y, z) in zip(barcodes, lattice):
        query = np.array([x, y]).reshape(2, 1)
        query /= config.phylogeny.spatial_scale

        isin = [clone.ellipse.contains(query) for clone in clones]

        candidates = [clone for clone, inside in zip(clones, isin) if inside]

        # NB we choose the smallest of overlapping ellipse as a (close) proxy for later evolved.
        if candidates:
            matched = min(candidates, key=lambda c: c.ellipse.det_l)
            cnas = matched.cnas
        else:
            cnas = []

        umis = 10.0 ** np.random.normal(
            loc=config.visium.log10umi_per_spot, scale=config.visium.log10umi_std_per_spot
        )

        snp_umis = 10.0 ** np.random.normal(
            loc=config.visium.log10snp_umi_per_spot, scale=config.visium.log10snp_umi_std_per_spot
        )

        mat_copies = np.ones(num_segments)
        pat_copies = np.ones(num_segments)
        
        if cnas:
            for cna in cnas:
                pos_idx = int(np.floor(cna[1] / config.segment_size_kbp))
                state = cna[0]

                mat_copy, pat_copy = cna[0].split(",")
                
                mat_copies[pos_idx] = int(mat_copy)
                pat_copies[pos_idx] = int(pat_copy)
                                
        print(f"{bc}\t{x}\t{y}\t{z}\t{umis:.3f}\t{snp_umis:.3f}")
            
    exit(0)

    logger.info(f"Generated visium to {sample_dir}")
