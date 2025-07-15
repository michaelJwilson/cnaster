import os
import gzip
import glob
import logging
import pandas as pd
import numpy as np
from numba import njit
from pathlib import Path
from cnaster.sim.clone import (
    Clone,
    get_clones,
    query_clones,
    construct_frac_cnas,
    get_cnas,
)
from cnaster.sim.io import get_exp_baseline, get_snp_baseline
from cnaster_rs import get_triangular_lattice, sample_segment_umis

logger = logging.getLogger(__name__)


def generate_fake_barcodes(num_spots):
    return [f"VIS{i:05d}" for i in range(num_spots)]


def assign_counts_to_segments(total, weights):
    num_segments = len(weights)
    choices = np.random.choice(num_segments, size=int(round(total)), p=weights)

    # TODO HACK
    return np.bincount(choices, minlength=1 + num_segments)


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

    num_segments = config.mappable_genome_kbp // config.segment_size_kbp

    segment_exp_baseline = get_exp_baseline(config)
    snps_segment = get_snp_baseline(config)

    meta = pd.DataFrame(
        {
            "barcode": pd.Series(dtype="str"),
            "umis": pd.Series(dtype=int),
            "snp_umis": pd.Series(dtype=int),
        }
    )

    truth = pd.DataFrame(
        {
            "barcode": pd.Series(dtype="str"),
            "clone": pd.Series(dtype=int),
            "tumor_purity": pd.Series(dtype=float),
        }
    )

    # TODO HARDCODE phlogeny id.
    clones = get_clones(config)

    # NB transcript umis and b-allele umis for all sports and segments.
    result = np.zeros(
        shape=(2, config.visium.nx * config.visium.ny, num_segments), dtype=float
    )

    for ii, (bc, (x, y, z)) in enumerate(zip(barcodes, lattice)):
        matched = query_clones(config, clones, x, y, z)
        cnas = matched.cnas if matched is not None else ()

        tumor_purity = 0.0

        if cnas:
            mean_purity = config.phylogeny.mean_purity
            tumor_purity = mean_purity + (1.0 - mean_purity) * np.random.uniform()

        rdrs, bafs = construct_frac_cnas(
            num_segments, config.segment_size_kbp, tumor_purity, cnas
        )

        # NB sample coverages for the spot
        total_umis = 10.0 ** np.random.normal(
            loc=config.visium.log10umi_per_spot,
            scale=config.visium.log10umi_std_per_spot,
        )

        total_snp_umis = 10.0 ** np.random.normal(
            loc=config.visium.log10snp_umi_per_spot,
            scale=config.visium.log10snp_umi_std_per_spot,
        )

        segment_baseline_umis = assign_counts_to_segments(
            total_umis, segment_exp_baseline
        )

        weights = segment_exp_baseline * snps_segment
        weights /= weights.sum()

        segment_baseline_snp_umis = assign_counts_to_segments(
            total_snp_umis,
            weights,
        )

        # NB assumes a single pseudo-count.
        ps = np.random.beta(
            1.0 + config.baf_dispersion * bafs,
            1.0 + config.baf_dispersion * (1.0 - bafs),
        )

        result[0, ii, :] = sample_segment_umis(
            segment_baseline_umis, rdrs, config.rdr_over_dispersion
        )
        result[1, ii, :] = np.random.binomial(segment_baseline_snp_umis, ps)

        meta_row = {
            "barcode": bc,
            "umis": int(total_umis),
            "snp_umis": int(total_snp_umis),
        }

        truth_row = {
            "barcode": bc,
            "clone": matched.id if matched is not None else -1,
            "tumor_purity": tumor_purity,
        }

        meta = pd.concat([meta, pd.DataFrame([meta_row])], ignore_index=True)
        truth = pd.concat([truth, pd.DataFrame([truth_row])], ignore_index=True)

    opath = Path(sample_dir) / "meta" / f"{name}.tsv.gz"
    opath.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing metadata to {str(opath)}")

    with gzip.open(opath, "wt") as f:
        f.write(f"# {meta.columns.to_list()}\n")

        meta.to_csv(
            f,
            sep="\t",
            index=False,
            header=False,
        )

    opath = Path(sample_dir) / "truth" / f"{name}.tsv.gz"
    opath.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing truthdata to {str(opath)}")

    with gzip.open(opath, "wt") as f:
        f.write(f"# {truth.columns.to_list()}\n")

        truth.to_csv(
            f,
            sep="\t",
            index=False,
            header=False,
        )

    opath = Path(sample_dir) / f"{name}_umis.npy"

    logger.info(f"Writing umis to {str(opath)}")

    np.save(opath, result)

    logger.info(f"Generated visium to {sample_dir}")
