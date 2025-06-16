import os
import gzip
import glob
import logging
import pandas as pd
import numpy as np
from numba import njit
from pathlib import Path
from cnaster.sim.clone import Clone
from cnaster.sim.io import get_exp_baseline, get_snp_baseline
from cnaster_rs import get_triangular_lattice, sample_segment_umis

logger = logging.getLogger(__name__)


def generate_fake_barcodes(num_spots):
    return [f"VIS{i:05d}" for i in range(num_spots)]


def assign_counts_to_segments(total, weights):
    num_segments = len(weights)
    choices = np.random.choice(
        num_segments, size=int(round(total)), p=weights
    )

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

    x0 = np.array([0.5, 0.5]).reshape(2, 1)

    # TODO HARDCODE phylogeny2
    clones = [
        Clone(xx, x0=x0)
        for xx in sorted(
            glob.glob(config.output_dir + f"/phylogenies/phylogeny4/*.json")
        )
    ]

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

    # NB transcript umis and b-allele umis for all sports and segments.
    result = np.zeros(shape=(2, config.visium.num_spots, num_segments), dtype=float)

    for ii, (bc, (x, y, z)) in enumerate(zip(barcodes, lattice)):
        # NB find the corresponding clone.
        query = np.array([x, y]).reshape(2, 1)
        query /= config.phylogeny.spatial_scale

        isin = [clone.ellipse.contains(query) for clone in clones]

        candidates = [clone for clone, inside in zip(clones, isin) if inside]

        # NB we choose the smallest of overlapping ellipse as a (close) proxy for later evolved.
        if candidates:
            matched = min(candidates, key=lambda c: c.ellipse.det_l)
            cnas = matched.cnas
        else:
            matched = None
            cnas = []

        tumor_purity = 0.0

        rdrs = np.ones(num_segments, dtype=float)
        bafs = 0.5 * np.ones(num_segments, dtype=float)

        # NB compute the purity, rdrs and bafs for this spot.
        for cna in cnas:
            pos_idx = int(np.floor(cna[1] / config.segment_size_kbp))
            state = cna[0]

            mat_copy, pat_copy = [int(xx) for xx in cna[0].split(",")]

            rdr = (mat_copy + pat_copy) / 2
            baf = min([mat_copy, pat_copy]) / (mat_copy + pat_copy)

            mean_purity = config.phylogeny.mean_purity
            tumor_purity = mean_purity + (1.0 - mean_purity) * np.random.uniform()

            rdrs[pos_idx] = (1. - tumor_purity) + (rdr * tumor_purity)
            bafs[pos_idx] = 0.5 * (1. - tumor_purity) + tumor_purity * baf * rdr

        # NB sample coverages for the spot
        total_umis = 10.0 ** np.random.normal(
            loc=config.visium.log10umi_per_spot,
            scale=config.visium.log10umi_std_per_spot,
        )

        total_snp_umis = 10.0 ** np.random.normal(
            loc=config.visium.log10snp_umi_per_spot,
            scale=config.visium.log10snp_umi_std_per_spot,
        )

        # NB input args:
        #     -  config.segment_size_kbp
        #     -  baseline exp. per segment
        #     -  snps per segment
        #     -  tumor_purity
        #     -  [segment_idx, rdr, baf] for all CNAs, if any.

        # TODO:
        #     - generate realized umis per segment as negative binomial.
        #     - generate realized snp umis per segment as beta_binomial.
        #
        # RETURN:
        #     - Vector of spot realized umis per segment, spot realized b-allele umis per segment.

        segment_baseline_umis = assign_counts_to_segments(
            total_umis, segment_exp_baseline
        )

        weights = segment_exp_baseline * snps_segment
        weights /= weights.sum()

        segment_baseline_snp_umis = assign_counts_to_segments(
            total_snp_umis, weights,
        )

        result[0, ii, :] = sample_segment_umis(segment_baseline_umis, rdrs, config.rdr_over_dispersion)

        print(ii)

        exit(0)

        """
        for ii in range(num_segments):
            # TODO HACK
            pp = np.random.beta(
                1.0 + config.baf_dispersion * baf,
                1.0 + config.baf_dispersion * (1.0 - baf),
            )
            
            segment_b = np.random.binomial(baseline_snp_umis[ii], pp)

            # print(bc, ii, segment_umi, segment_b, config.baf_dispersion, bafs[ii])
        """

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

    logger.info(f"Generated visium to {sample_dir}")