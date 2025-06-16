import os
import gzip
import glob
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from cnaster.sim.clone import Clone
from cnaster_rs import get_triangular_lattice

logger = logging.getLogger(__name__)


def generate_fake_barcodes(num_spots):
    return [f"VIS{i:05d}" for i in range(num_spots)]


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
    data = np.zeros(shape=(2, config.visium.num_spots, num_segments), dtype=float)

    for bc, (x, y, z) in zip(barcodes, lattice):
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
            matched=None
            cnas = []

        tumor_purity = 0.

        # NB compute the purity, rdrs and bafs for this spot.
        for cna in cnas:
            pos_idx = int(np.floor(cna[1] / config.segment_size_kbp))
            state = cna[0]

            mat_copy, pat_copy = [int(xx) for xx in cna[0].split(",")]

            rdr = (mat_copy + pat_copy) / 2
            baf = min([mat_copy, pat_copy]) / (mat_copy + pat_copy)

            mean_purity = 0.75
            tumor_purity = mean_purity + (1. - mean_purity) * np.random.uniform()

        # NB sample coverages for the spot
        umis = 10.0 ** np.random.normal(
            loc=config.visium.log10umi_per_spot,
            scale=config.visium.log10umi_std_per_spot,
        )

        snp_umis = 10.0 ** np.random.normal(
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
        #     - generate baseline umis per gene as Poisson(umis / num_genes)
        #     - generate baseline umis per segment by aggregating baseline umis per gene by num. genes per segment
        #     - generate baseline snp umis per snp as Poisson(snp_umis / num_snps)
        #     - generate baseline snp umis per segment by aggregating baseline umis per snp by num. snps per segment
        #     - generate realized umis per segment as negative binomial.
        #     - generate realized snp umis per segment as beta_binomial.
        # 
        # RETURN:
        #     - Vector of spot realized umis per segment, spot realized b-allele umis per segment.

        """
        # NB genes and snps are non-uniformly distributed across segments.
        # TODO runs slow
        num_snps_segments = np.random.poisson(lam=exp_snps_segment, size=num_segments)
        
        # TODO no constraint that snp_coverage < coverage
        baseline_segment_umis = np.random.poisson(
            lam=umis / num_segments, size=num_segments
        )

        # NB depends on global number of snps across all segments.
        baseline_snp_umis = np.random.poisson(
            lam=snp_umis / num_snps_segments.sum(), size=num_snps_segments.sum()
        )
        
        idx = 0
        baseline_segment_snp_umis = np.zeros(num_segments, dtype=int)

        for seg_idx, n_snps in enumerate(num_snps_segments):
            if n_snps > 0:
                baseline_segment_snp_umis[seg_idx] = np.sum(
                    baseline_snp_umis[idx : idx + n_snps]
                )

                idx += n_snps
            else:
                baseline_segment_snp_umis[seg_idx] = 0

        
        for ii in range(num_segments):
            # NB accounts for tumor purity.
            rdr = (1. - tumor_purity) + (rdrs[ii] * tumor_purity)
            baf = 0.5 * (1. - tumor_purity) + tumor_purity * bafs[ii] * rdrs[ii]
            
            rr = 1.0 / config.rdr_over_dispersion
            pp = 1.0 / (
                1.0 + config.rdr_over_dispersion * rdr * baseline_segment_umis[ii]
            )

            segment_umi = np.random.negative_binomial(n=rr, p=pp)

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
            "umis":  int(umis),
            "snp_umis":   int(snp_umis),
        }

        truth_row = {
            "barcode": bc,
            "clone":  matched.id if matched is not None else -1,
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