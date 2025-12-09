import copy
import logging
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import polars as pl
import scipy.sparse
from collections import namedtuple
from cnaster.filter import get_filter_genes, get_filter_ranges
from cnaster.reference import exp_cancer_gene
from cnaster.config import get_global_config
from cnaster.utils import cacher
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)

pl.Config.set_tbl_cols(-1)


def get_sample_sheet(sample_sheet_path):
    df_meta = pd.read_csv(sample_sheet_path, sep=r"\s+")

    required_columns = {"bam", "sample_id", "spaceranger_dir", "snp_dir"}

    # TODO
    assert required_columns.issubset(
        df_meta.columns
    ), f"sample_sheet has columns {df_meta.columns} which missese the required: {required_columns - set(df_meta.columns)}:\n{df_meta}"

    logger.info(
        f"Input sample_sheet_path={sample_sheet_path} contains {len(df_meta)} samples:\n{df_meta}"
    )

    return df_meta


# TODO check (e.g. sample sheet with john): AAACAAGTATCTCCCA-1_HT112C1-U1 == {spot}-1_{sample_id}-{slice}.
def get_aggregated_barcodes(barcode_file, known_sample_id=None):
    # NB see https://github.com/raphael-group/CalicoST/blob/5e4a8a1230e71505667d51390dc9c035a69d60d9/calicost.smk#L32
    df_barcode = pd.read_csv(barcode_file, header=None, names=["combined_barcode"])
    sample_id_defined = df_barcode.combined_barcode.str.contains("_").all()

    # NB per-slice Visium 10x defined barcode.
    df_barcode["barcode"] = [
        x.split("_")[0] for x in df_barcode.combined_barcode.to_numpy()
    ]

    if sample_id_defined:
        logger.info(f"Found defined sample_ids")
        df_barcode["sample_id"] = [
            x.split("_")[-1] for x in df_barcode.combined_barcode.to_numpy()
        ]
    else:
        logger.warning(
            f"Unable to resolve sample_ids from aggregated barcodes.  Assuming known sample_id={known_sample_id}."
        )
        df_barcode["sample_id"] = (
            known_sample_id if known_sample_id is not None else "UNKNOWN"
        )

    # TODO HACK
    df_barcode["barcode"] = df_barcode.combined_barcode
    df_barcode["sample_id"] = known_sample_id

    # TODO sample ids currently slice, e.g. U1;
    logger.info(
        f"Input aggregated barcode file {barcode_file} with {df_barcode.shape[0]:_} barcodes for all samples/bams, e.g.\n{df_barcode.head()}\n"
    )

    return df_barcode


def get_spatial_positions(spaceranger_dir, filter_in_tissue=True):
    """ """
    # TODO x,y vs row,col?  sub-pixel position?
    names = ("barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col")

    if Path(
        f"{spaceranger_dir}/spatial/tissue_positions.csv",
    ).exists():
        df_this_pos = pd.read_csv(
            f"{spaceranger_dir}/spatial/tissue_positions.csv",
            sep=",",
            header=0,
            names=names,
        )

        logger.info(f"Reading {spaceranger_dir}/spatial/tissue_positions.csv")

    elif Path(f"{spaceranger_dir}/spatial/tissue_positions_list.csv").exists():
        df_this_pos = pd.read_csv(
            f"{spaceranger_dir}/spatial/tissue_positions_list.csv",
            sep=",",
            header=None,
            names=names,
        )

        logger.info(f"Reading {spaceranger_dir}/spatial/tissue_positions_list.csv")

    elif Path(f"{spaceranger_dir}/spatial/tissue_positions.parquet").exists():
        # NB 11,222,500 rows vs 4,992 rows for visium.
        #    see https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs
        #
        #    native columns:  barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres.

        """
        df_this_pos = (
            pl.scan_parquet(f"{spaceranger_dir}/spatial/tissue_positions.parquet")
            .rename({"pxl_row_in_fullres": "y", "pxl_col_in_fullres": "x"})
            .select(["barcode", "in_tissue", "x", "y"])
            # .filter(pl.col("in_tissue") == True)
            .collect()
            .with_columns(pl.col("barcode").alias("square_002um"))
        )

        # NB native columns: square_002um, square_008um, square_016um, cell_id, in_nucleus, in_cell
        # TODO CHECK in_cell
        df_this_pos = (
            pl.scan_parquet(f"{spaceranger_dir}/spatial/barcode_mappings.parquet")
            .filter(pl.col("in_cell") == True)
            .collect()
            .join(
                df_this_pos.select(["square_002um", "x", "y", "in_tissue"]),
                on="square_002um",
                how="left",
            )
            .select(["square_002um", "cell_id", "x", "y", "in_tissue"])
            .group_by("cell_id")
            .agg(
                [
                    pl.col("x").mean().alias("x"),
                    pl.col("y").mean().alias("y"),
                    pl.col("square_002um").n_unique().alias("num_square_002um"),
                    pl.col("in_tissue").cast(pl.Boolean).any().alias("in_tissue"),
                ]
            )
            .with_columns(pl.col("cell_id").alias("barcode"))
        )
        """
        df_this_pos = (
            pl.scan_parquet(f"{spaceranger_dir}/spatial/tissue_positions.parquet")
            .rename({"pxl_row_in_fullres": "y", "pxl_col_in_fullres": "x"})
            .filter(pl.col("in_tissue") == True)
            .collect()
        )

        logger.info(
            f"Read {spaceranger_dir}/spatial/tissue_positions.parquet:\n{df_this_pos}"
        )

        df_this_pos = df_this_pos.to_pandas()  # .set_index("barcode")

    else:
        logger.error(f"No spatial coordinate file @ {spaceranger_dir}.")
        raise RuntimeError()

    # TODO alignment defined for in_tissue == True only?
    if filter_in_tissue:
        logger.warning(
            f"Filtering spatial positions to in_tissue == True (retained {100. * np.mean(df_this_pos.in_tissue)}%)."
        )

        result = df_this_pos[df_this_pos.in_tissue == True]
    else:
        result = df_this_pos

    # NB x,y positions for each barcode in this sample.
    return result


def get_spaceranger_counts(spaceranger_dir):
    config = get_global_config()
    filtered_feature_name = config.visium.filtered_feature_name

    supported_types = ["filtered_feature_bc_matrix", "filtered_feature_cell_matrix"]

    if filtered_feature_name not in supported_types:
        logger.error(
            f"{filtered_feature_name} is not supported; expected one  of {supported_types}"
        )
        raise ValueError()

    # NB see https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_10x_h5.html
    if Path(f"{spaceranger_dir}/{filtered_feature_name}.h5").exists():
        adatatmp = sc.read_10x_h5(
            f"{spaceranger_dir}/{filtered_feature_name}.h5",  # gex_only=True
        )
        logger.info(f"Reading {spaceranger_dir}/{filtered_feature_name}.h5")

    elif Path(f"{spaceranger_dir}/{filtered_feature_name}.h5ad").exists():
        adatatmp = sc.read_h5ad(
            f"{spaceranger_dir}/{filtered_feature_name}.h5ad",  # gex_only=True
        )
        logger.info(f"Reading {spaceranger_dir}/{filtered_feature_name}.h5ad")

    else:
        logging.error(
            f"{spaceranger_dir} directory does not have a {filtered_feature_name}.h5(ad)!"
        )
        raise RuntimeError()

    # TODO comment on adatatmp.x (nobs x nvars for space ranger, i.e. barcodes x gene transcripts).
    adatatmp.layers["count"] = adatatmp.X.toarray()

    is_nan = np.isnan(adatatmp.layers["count"])

    logger.info(f"Found {100.0 * np.mean(is_nan):.3f}% NaN counts in anndata.")

    # NB replace nan with 0 and cast to int.
    if np.any(is_nan):
        adatatmp.layers["count"][is_nan] = 0

    # TODO CHECK
    adatatmp.layers["count"] = adatatmp.layers["count"].astype(int)

    # e.g. duplicated:  TBCE  2, LINC01238  2.3; why?
    # duplicated_mask = adatatmp.var_names.duplicated(keep=False)
    # non_unique_vars = adatatmp.var_names[duplicated_mask]

    # duplicate_counts = non_unique_vars.value_counts()

    logger.info(
        f"Read transcript counts of shape {adatatmp.shape}, i.e. (barcodes, genes) from {spaceranger_dir}"
    )

    logger.info(
        f"Example names for {len(adatatmp.obs_names):_} barcodes:\n{adatatmp.obs_names[:5]}"
    )
    logger.info(
        f"Example names for {len(adatatmp.var_names):_} genes:\n{adatatmp.var_names[:5]}"
    )

    # NB var names made unique by appending an index string,
    #    see https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.var_names_make_unique.html
    adatatmp.var_names_make_unique()

    # NB data matrix X (ndarray/csr matrix, dask ...): observations/cells are named by their barcode and variables/genes by gene name.
    return adatatmp


# TODO massively inefficient?
# NB mirrors https://github.com/raphael-group/CalicoST/blob/c1abcae3e3657e01e547ee4529e3b9d039221453/src/calicost/utils_IO.py#L127
def get_alignments(alignment_files, df_meta, df_agg_barcode, significance=1.0e-6):
    if alignment_files is None:
        return None

    row_ind, col_ind = [], []
    dat = []

    offset = 0

    for i, f in enumerate(alignment_files):
        pi = np.load(f)

        # normalize p such that max( row_sums(pi), cols_sum(pi) ) = 1;
        # TODO? max alignment weight = 1
        pi = pi / np.max(np.append(np.sum(pi, axis=0), np.sum(pi, axis=1)))

        # NB assumes alignments ordered by df_meta sample_ids.
        sname1 = df_meta.sample_id.to_numpy()[i]
        sname2 = df_meta.sample_id.to_numpy()[i + 1]

        assert pi.shape[0] == np.sum(df_agg_barcode["sample_id"] == sname1)
        assert pi.shape[1] == np.sum(df_agg_barcode["sample_id"] == sname2)

        # NB for each spot s in sname1, select {t: spot t in sname2 and pi[s,t] >= np.max(pi[s,:])} as the corresponding spot in the other slice
        for row in range(pi.shape[0]):
            row_max = np.max(pi[row, :])

            # NB their exists an element in the alignment of a sample 1 spot with significant probability (> significance)
            cutoff = row_max if row_max > significance else 1.0 + significance

            list_cols = np.where(pi[row, :] >= cutoff - significance)[0]

            row_ind += [offset + row] * len(list_cols)

            # NB zero_point = offset + pi.shape[0] +1 per col entry.
            col_ind += list(offset + pi.shape[0] + list_cols)

            dat += list(pi[row, list_cols])

        offset += pi.shape[0]

    across_slice_adjacency_mat = scipy.sparse.csr_matrix(
        (dat, (row_ind, col_ind)), shape=(adata.shape[0], adata.shape[0])
    )

    # TODO symmetric by definition.
    across_slice_adjacency_mat += across_slice_adjacency_mat.T

    return across_slice_adjacency_mat


def map_unique_snps_enum(unique_snp_ids):
    """
    Given unique_snp_ids (array) of {contig}_{pos}_{ref}_{alt} for all snps,
    where ref = alt = N is a potentially anonymized snp (unknown base),
    map each snp to a unique id of the form {contig}_{pos}_{enum}, where enum
    allows for erroneous repeats, but is typically 0.
    """
    # NB log the number of unique snps and warn on any repeats
    bonafide_unique_snps, cnts = np.unique(unique_snp_ids, return_counts=True)
    logger.info(
        f"Detected {len(bonafide_unique_snps)} unique snps from {len(unique_snp_ids)} input snp ids with dtype={unique_snp_ids.dtype}."
    )

    repeats = dict()

    if len(bonafide_unique_snps) != len(unique_snp_ids):
        for snp_id, count in zip(bonafide_unique_snps[cnts > 1], cnts[cnts > 1]):
            contig, pos, _, _ = snp_id.split("_")
            logger.warning(
                f"Detected repeated snp_id @ chr{contig}:{pos} with count={count}."
            )
            repeats[snp_id] = 0

    result = []

    for snp_id in unique_snp_ids:
        contig, pos, _, _ = snp_id.split("_")

        if snp_id in repeats:
            enum = repeats[snp_id]
            repeats[snp_id] += 1
        else:
            enum = 0

        new_snp_id = f"{contig}_{pos}_{enum}"
        result.append(new_snp_id)

    result = np.array(result, dtype=unique_snp_ids.dtype)

    logger.info(f"Mapped input snp ids to enum:\n{result[:5]}")

    return result


@cacher("processed_input.hdf5")
def load_input_data(
    config,
    alignment_files=None,
    filter_gene_file=None,
    filter_range_file=None,
    normal_idx_file=None,
    min_snp_umis=50,
    min_percent_expressed_spots=5.0e-3,  # BUG actually a fraction.
):
    if alignment_files is None:
        logger.warning(f"Alignment files not provided")
    elif len(alignment_files) + 1 != df_meta.shape[0]:
        logger.error(f"Incorrect number of alignment files found.")
        raise RuntimeError()
    else:
        raise NotImplementedError("Alignment files are not supported.")

    # NB see https://github.com/raphael-group/CalicoST/blob/5e4a8a1230e71505667d51390dc9c035a69d60d9/src/calicost/utils_IO.py#L127
    df_meta = get_sample_sheet(config.paths.sample_sheet)

    # NB assumes snps derived from aggregation of all provided samples,
    assert np.all(df_meta["snp_dir"] == df_meta["snp_dir"].iloc[0])

    # NB (phased) SNPs are determined for the pseudobulk of all spots.
    snp_dir = df_meta["snp_dir"].iloc[0]

    # TODO sample_id not defined?  barcodes uniquely identify each spot per slice,
    #      aggregated across slices/bams.
    known_sample_id = df_meta.sample_id[0] if len(df_meta) == 1 else None
    df_agg_barcode = get_aggregated_barcodes(f"{snp_dir}/barcodes.txt", known_sample_id)

    # TODO duplicate of df_agg_barcode
    # NB dataframe of combined barcodes, i.e. Visium barcode + slice 'sample_id'.
    snp_barcodes = pd.read_csv(
        f"{snp_dir}/barcodes.txt", header=None, names=["barcodes"]
    )

    """
    # TODO HACK >>>>>>>>
    try:
        sample_id_patcher = {
            sample_id.split("-")[1]: sample_id for sample_id in df_meta.sample_id.to_numpy()
        }

        df_agg_barcode["sample_id"] = df_agg_barcode["sample_id"].map(sample_id_patcher)

        snp_barcodes["barcodes"] = snp_barcodes["barcodes"].map(
            lambda xx: xx.split("_")[0] + "_" + sample_id_patcher[xx.split("_")[-1]]
        )
    except:
        logger.warning(f"Failed to patch input sample ids.")
    # <<<<<<<<<
    """

    unique_snp_ids = np.load(f"{snp_dir}/unique_snp_ids.npy", allow_pickle=True)
    unique_snp_ids = map_unique_snps_enum(unique_snp_ids)

    # NB read (phased) counts for H0/H1 for (spots, snps).
    cell_snp_Aallele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Aallele.npz")
    cell_snp_Ballele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Ballele.npz")

    assert cell_snp_Aallele.shape == cell_snp_Ballele.shape

    cell_snp = (cell_snp_Aallele + cell_snp_Ballele).todense().sum(axis=1)

    logger.info(
        f"Read cell-snp A,B matrices of shape={cell_snp_Aallele.shape} with min={cell_snp.min()}, max={cell_snp.max()}, median={np.median(cell_snp[0])} snp-umis per cell.  Found {np.sum(cell_snp):_} snp-umis total."
    )

    # NB read Visium transcripts/UMIs anndata & spot spatial coordinate.
    adata = None

    # NB df_meta provides the sample_ids, one per bam.
    for i, sname in enumerate(df_meta.sample_id.to_numpy()):
        logger.info(f"Reading (spot, gene) UMIs for spaceranger sample={sname}.")

        index = np.where(df_agg_barcode["sample_id"] == sname)[0]

        logger.info(f"Found {len(index)}/{len(df_agg_barcode)} matches by sample_id.")

        # NB indexed spot barcodes for this sample/slice.
        df_this_barcode = copy.copy(df_agg_barcode.iloc[index, :])
        df_this_barcode.index = df_this_barcode.barcode

        # NB (x,y) positions for each barcode (one per row).  limited to "in tissue" by default.
        df_this_pos = get_spatial_positions(df_meta["spaceranger_dir"].iloc[i])

        # NB read filtered_feature_bc_matrix.h5(ad) from spaceranger_dir for this sample - UMIs (spot barcode, gene).
        adatatmp = get_spaceranger_counts(df_meta["spaceranger_dir"].iloc[i])

        # NB re-order anndata spots to have the order of "df_this_barcode" (with enum).
        idx_argsort = pd.Categorical(
            adatatmp.obs.index, categories=list(df_this_barcode.barcode), ordered=True
        ).argsort()

        if not np.array_equal(idx_argsort, np.arange(len(idx_argsort))):
            logger.info(f"Sorting ST data by barcode.")
            adatatmp = adatatmp[idx_argsort, :].copy()

        # NB only keep shared barcodes between (IN_TISSUE) visium barcodes and filtered_feature_bc_matrix.
        pos_barcodes = set(list(df_this_pos.barcode))
        count_barcodes = set(list(adatatmp.obs.index))

        shared_barcodes = pos_barcodes & count_barcodes

        isin = adatatmp.obs.index.isin(shared_barcodes)

        logger.info(
            f"Retaining {100.0 * np.mean(isin):.3f}% of spots based on (in-tissue) position and UMIs."
        )

        # TODO visium hd.
        if not isin.all():
            adatatmp = adatatmp[isin, :].copy()

        df_this_pos = df_this_pos[df_this_pos.barcode.isin(shared_barcodes)]

        # NB re-order positions to have order of df_this_barcode barcodes.
        df_this_pos.barcode = pd.Categorical(
            df_this_pos.barcode, categories=list(adatatmp.obs.index), ordered=True
        )

        df_this_pos.sort_values(by="barcode", inplace=True)

        adatatmp.obsm["X_pos"] = np.vstack([df_this_pos.x, df_this_pos.y]).T
        adatatmp.obs["sample"] = sname

        # NB index by {barcode}_{sample} (TBC)
        # adatatmp.obs.index = [f"{x}_{sname}" for x in adatatmp.obs.index]

        # TODO HACK
        adatatmp.obs.index = [f"{x}" for x in adatatmp.obs.index]

        # NB concatenate across samples.
        adata = (
            adatatmp
            if adata is None
            else anndata.concat([adata, adatatmp], join="outer")
        )

    # NB filter by spots:  shared barcodes between adata and SNPs; e.g. drop spots with SNP counts but no transcripts.
    shared_barcodes = set(list(snp_barcodes.barcodes)) & set(list(adata.obs.index))

    isin = snp_barcodes.barcodes.isin(shared_barcodes).to_numpy()

    # TODO barcode inconsistent between snps and umis.
    assert np.any(
        isin
    ), f"Found inconsistent barcodes between SNPs and UMIs, e.g. {list(snp_barcodes.barcodes)[:5]} vs {list(adata.obs.index)[:5]}"

    logger.info(
        f"Retaining {100.0 * np.mean(isin):.3f}% of SNP barcodes (shared between UMIs and SNPs)."
    )

    # NB barcode (row) selection.
    if not isin.all():
        cell_snp_Aallele = cell_snp_Aallele[isin, :]
        cell_snp_Ballele = cell_snp_Ballele[isin, :]

        snp_barcodes = snp_barcodes[isin]

    isin = adata.obs.index.isin(shared_barcodes)

    logger.info(
        f"Retaining {100.0 * np.mean(isin):.3f}% of UMI barcodes (shared between UMIs and SNPs)."
    )

    if not isin.all():
        adata = adata[isin, :].copy()

    idx_argsort = pd.Categorical(
        adata.obs.index, categories=list(snp_barcodes.barcodes), ordered=True
    ).argsort()

    if not np.array_equal(idx_argsort, np.arange(len(idx_argsort))):
        logger.info(f"Sorting data by barcode.")
        adata = adata[idx_argsort, :]

    across_slice_adjacency_mat = get_alignments(
        alignment_files, df_meta, df_agg_barcode
    )

    # NB filter out spots with too small number of UMIs (genome wide);
    # TODO differentiate min_snpumis; why before genomic binning?
    indicator = np.sum(adata.layers["count"], axis=1) >= min_snp_umis

    logger.info(
        f"Retaining {100.0 * np.mean(indicator):.3f}% of spots with sufficient UMIs (>= {min_snp_umis})."
    )

    # NB retain barcodes with sufficient SNP covering UMIs per spot.
    indicator &= (
        np.sum(cell_snp_Aallele, axis=1).A.flatten()
        + np.sum(cell_snp_Ballele, axis=1).A.flatten()
        >= min_snp_umis
    )

    logger.info(
        f"Retaining {100.0 * np.mean(indicator):.3f}% of spots with sufficient snp-UMIs (>= {min_snp_umis})."
    )

    adata = adata[indicator, :]

    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]

    if across_slice_adjacency_mat is not None:
        across_slice_adjacency_mat = across_slice_adjacency_mat[indicator, :][
            :, indicator
        ]

    # TODO HACK
    logger.info(f"Found total UMI = {np.sum(adata.layers['count']):_} in input data.")

    spot_umis = np.sum(adata.layers["count"], axis=1)
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    perc_vals = np.percentile(spot_umis, percentiles)

    pairs = "\n".join(
        f"{p:.3f} [%]\t{v:_.0f}" for p, v in zip(percentiles, perc_vals)
    )
    logger.info(f"UMIs per spot percentiles:\n{pairs}")

    # NB filter out genes that are expressed in < min_percent_expressed_spots spots.
    indicator = (
        # NB number of barcodes expressing a particular gene; num. spots.
        np.sum(adata.X > 0, axis=0)
        >= min_percent_expressed_spots * adata.shape[0]
    ).A.flatten()

    # NB ratio of total UMIs across all spots for gene selection vs all.
    ratio = np.sum(adata.X[:, indicator]) / np.sum(adata.X)

    # TODO gencode gene list is not all sampled by (3') visium umis.
    # TODO excludes 50% of genes, but retains 99.97% of UMIs; resolves gene definition to house-keeping?
    logger.info(
        f"Retaining {100.0 * np.mean(indicator):.3f}% of genes with sufficient expression across spots ({100.0 * ratio:.2f}% of total UMIs) @ {min_percent_expressed_spots} fraction of spots."
    )

    adata = adata[:, indicator]

    logger.info(
        f"Median spot UMI after filtering genes based on num. spots expressed = {np.median(np.sum(adata.layers['count'], axis=1)):_.3f}"
    )

    if filter_gene_file is not None:
        genes_to_filter = get_filter_genes(filter_gene_file).iloc[:, 0].to_numpy()
        indicator_filter = ~np.isin(adata.var.index, genes_to_filter)

        logger.info(
            f"Removing {len(filter_gene_file)} genes based on input file={filter_gene_file}."
        )

        # for to_print in genes_to_filter[np.isin(genes_to_filter, adata.var.index)]:
        #   logger.info(to_print)

        adata = adata[:, indicator_filter]

        logger.info(
            f"Median spot UMI after filtering genes = {np.median(np.sum(adata.layers['count'], axis=1)):_.3f}"
        )

        # TODO?
        # apply ranges cut to cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids?

    if filter_range_file is not None:
        ranges = get_filter_ranges(filter_range_file)
        num_ranges = ranges.shape[0]

        # NB defaults to retain all SNP counts, excluded based on filter_range_file.
        indicator_filter = np.array([True] * cell_snp_Aallele.shape[1])
        j = 0

        # TODO read-through / slow.
        for i in range(cell_snp_Aallele.shape[1]):
            this_chr = int(unique_snp_ids[i].split("_")[0])
            this_pos = int(unique_snp_ids[i].split("_")[1])

            # NB fast forward genomic position
            while j < num_ranges and (
                (ranges.Chr.to_numpy()[j] < this_chr)
                or (
                    (ranges.Chr.to_numpy()[j] == this_chr)
                    and (ranges.End.to_numpy()[j] <= this_pos)
                )
            ):
                j += 1

            if (
                j < num_ranges
                and (ranges.Chr.to_numpy()[j] == this_chr)
                and (ranges.Start.to_numpy()[j] <= this_pos)
                and (ranges.End.to_numpy()[j] > this_pos)
            ):
                indicator_filter[i] = False

        logger.info(
            f"Retaining {100.0 * np.mean(indicator_filter):.2f}% of SNPs based on input filter ranges."
        )

        cell_snp_Aallele = cell_snp_Aallele[:, indicator_filter]
        cell_snp_Ballele = cell_snp_Ballele[:, indicator_filter]

        unique_snp_ids = unique_snp_ids[indicator_filter]

    if config.quality.local_outlier_filter:
        # NB  k-NN defined density estimates used to filter local outliers given density wrt neighbors.
        #     see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
        #         https://en.wikipedia.org/wiki/Local_outlier_factor
        clf = LocalOutlierFactor(n_neighbors=200)

        # NB  prediction on barcode summed transcripts for each gene.
        label = clf.fit_predict(np.sum(adata.layers["count"], axis=0).reshape(-1, 1))

        to_zero = np.where(label == -1)[0]

        # NB ratio of total UMIs across all spots for gene selection vs all.
        ratio = np.sum(adata.layers["count"][:, to_zero]) / np.sum(
            adata.layers["count"]
        )

        # TODO removed 235 outlier genes (51.310% of UMIs)!!
        logger.info(
            f"Removed {len(to_zero)} outlier genes ({100.0 * ratio:.3f}% of UMIs) based on {clf.__class__.__name__}."
        )

        if len(to_zero) > 0:
            # NB barcode summed counts per gene.
            gene_umi_counts = np.sum(adata.layers["count"], axis=0)
            total_umis = np.sum(adata.layers["count"])

            outlier_genes_info = []

            for gene_idx in to_zero:
                gene_name = adata.var.index[gene_idx]
                gene_umis = gene_umi_counts[gene_idx]
                gene_pct = 100.0 * gene_umis / total_umis
                outlier_genes_info.append((gene_name, gene_umis, gene_pct))

            outlier_genes_info.sort(key=lambda x: x[2], reverse=True)

            # TODO log chr, start, end.
            logger.info("Top 25 outlier genes removed:")

            for i, (gene_name, gene_umis, gene_pct) in enumerate(
                outlier_genes_info[:25]
            ):
                # NB altered mitochondrial metabolism (Warburg effect) in cancer;
                warning = (
                    "WARNING known to be cancerous"
                    if exp_cancer_gene(gene_name)
                    else ""
                )

                logger.info(
                    f"  {i+1:2d}. {gene_name:<20} {gene_pct:6.3f}% UMIs {warning}"
                )

        # NB  zero count of outlier genes (!)  Should retain snp-umi counts ...
        adata.layers["count"][:, to_zero] = 0

    elif config.quality.normalize_gene_outliers:
        PERCENTILE = 95

        gene_counts = np.sum(adata.layers["count"], axis=0)

        total_umis = np.sum(gene_counts)

        threshold = np.percentile(gene_counts, PERCENTILE)

        top_genes_indices = np.where(gene_counts >= threshold)[0]

        top_genes_umis = np.sum(gene_counts[top_genes_indices])

        target_umis = (1.0 - PERCENTILE / 100) * (total_umis - top_genes_umis)

        for gene_idx in top_genes_indices:
            current_umis = gene_counts[gene_idx]

            if current_umis > target_umis:
                downsampling_factor = target_umis / current_umis
                adata.layers["count"][:, gene_idx] = (
                    adata.layers["count"][:, gene_idx] * downsampling_factor
                )

        logger.info(
            f"Downsampled top {100. - PERCENTILE}% genes to ensure they contribute only {100. - PERCENTILE}% of final UMIs; originally {100. * top_genes_umis / total_umis:.3f} [%]."
        )

    if normal_idx_file is not None:
        normal_barcodes = (
            pd.read_csv(normal_idx_file, header=None).iloc[:, 0].to_numpy()
        )

        # NB column with tumor/normal designation.
        adata.obs["tumor_annotation"] = "tumor"
        adata.obs["tumor_annotation"][adata.obs.index.isin(normal_barcodes)] = "normal"

        logger.info(
            "Applied tumor annotation: {adata.obs['tumor_annotation'].value_counts()}"
        )

    logger.info(f"Realized AnnData:\n{adata}")

    # NB barcode consistency
    assert adata.layers["count"].shape[0] == cell_snp_Aallele.shape[0]
    assert cell_snp_Aallele.shape[0] == cell_snp_Ballele.shape[0]

    # NB SNP consistency; 17_797 anndata genes vs 16_681 SNPs.
    assert len(unique_snp_ids) == cell_snp_Aallele.shape[1]
    assert cell_snp_Aallele.shape[1] == cell_snp_Ballele.shape[1]

    ProcessedData = namedtuple(
        "ProcessedData",
        [
            "adata",
            "cell_snp_Aallele",
            "cell_snp_Ballele",
            "unique_snp_ids",
            "across_slice_adjacency_mat",
        ],
    )

    # TODO dense arrays.
    result = ProcessedData(
        adata,
        cell_snp_Aallele.toarray(),
        cell_snp_Ballele.toarray(),
        unique_snp_ids,
        across_slice_adjacency_mat,
    )

    return result
