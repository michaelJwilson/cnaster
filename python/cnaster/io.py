import copy
import logging
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from sklearn.neighbors import LocalOutlierFactor

from cnaster.filter import get_filter_genes, get_filter_ranges
from cnaster.omics import form_gene_snp_table
from cnaster.reference import get_reference_genes

logger = logging.getLogger(__name__)


def get_sample_sheet(sample_sheet_path):
    df_meta = pd.read_csv(sample_sheet_path, sep="\t", header=None)
    df_meta.rename(
        columns=dict(
            zip(df_meta.columns[:4], ["bam", "sample_id", "spaceranger_dir", "snp_dir"])
        ),
        inplace=True,
    )

    logger.info(f"Input sample_sheet_path={spaceranger_meta_path} contains:\n{df_meta}")

    return df_meta


def get_aggregated_barcodes(barcode_file):
    # NB see https://github.com/raphael-group/CalicoST/blob/5e4a8a1230e71505667d51390dc9c035a69d60d9/calicost.smk#L32
    df_barcode = pd.read_csv(barcode_file, header=None, names=["combined_barcode"])

    # NB per-slice Visium 10x defined barcode.
    df_barcode["barcode"] = [
        x.split("_")[0] for x in df_barcode.combined_barcode.values
    ]

    # NB user specified sample_id per bam.
    # TODO define sample_id if it does not exist.
    df_barcode["sample_id"] = [
        x.split("_")[-1] for x in df_barcode.combined_barcode.values
    ]

    logger.info(
        f"Input barcode file {barcode_file} with {df_barcode.shape[0]} barcodes for all samples/bams, e.g.\n{df_barcode.head()}\n"
    )

    return df_barcode


def get_spatial_positions(spaceranger_dir, filter_in_tissue=True):
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

        logger.info("Reading {spaceranger_dir}/spatial/tissue_positions.csv")

    elif Path(f"{spaceranger_dir}/spatial/tissue_positions_list.csv").exists():
        df_this_pos = pd.read_csv(
            f"{spaceranger_dir}/spatial/tissue_positions_list.csv",
            sep=",",
            header=None,
            names=names,
        )

        logger.info("Reading {spaceranger_dir}/spatial/tissue_positions_list.csv")

    else:
        logger.error("No spatial coordinate file @ {spaceranger_dir}.")
        raise RuntimeError()

    # TODO alignment defined for in_tissue == True only?
    if filter_in_tissue:
        return df_this_pos[df_this_pos.in_tissue == True]
    else:
        return df_this_pos


def get_spaceranger_counts(spaceranger_dir):
    # NB https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_10x_h5.html
    if Path(f"{spaceranger_dir}/filtered_feature_bc_matrix.h5").exists():
        adatatmp = sc.read_10x_h5(f"{spaceranger_dir}/filtered_feature_bc_matrix.h5")

        logger.info(f"Reading {spaceranger_dir}/filtered_feature_bc_matrix.h5")

    elif Path(f"{spaceranger_dir}/filtered_feature_bc_matrix.h5ad").exists():
        adatatmp = sc.read_h5ad(f"{spaceranger_dir}/filtered_feature_bc_matrix.h5ad")
        logger.info(f"Reading {spaceranger_dir}/filtered_feature_bc_matrix.h5ad")

    else:
        logging.error(
            f"{spaceranger_dir} directory does not have a filtered_feature_bc_matrix.h5(ad)!"
        )

        raise RuntimeError()

    # TODO
    adatatmp.layers["count"] = adatatmp.X.A

    # NB data matrix X (ndarray/csr matrix, dask ...): observations/cells are named by their barcode and variables/genes by gene name
    return adatatmp


# TODO massively inefficient?
def get_alignments(alignment_files, df_meta, df_agg_barcode, significance=1.0e-6):
    if len(alignment_files) == 0:
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
        sname1 = df_meta.sample_id.values[i]
        sname2 = df_meta.sample_id.values[i + 1]

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


def load_input_data(
    config,
    alignment_files=None,
    filter_gene_file=None,
    filter_range_file=None,
    normal_idx_file=None,
    min_snp_umis=50,
    min_percent_expressed_spots=5.0e-3,
    local_outlier_filter=True,
):
    # NB see https://github.com/raphael-group/CalicoST/blob/5e4a8a1230e71505667d51390dc9c035a69d60d9/src/calicost/utils_IO.py#L127
    df_meta = get_sample_sheet(config.paths.sample_sheet)

    # TODO HACK
    assert len(df_meta) == 1

    snp_dir = df_meta["snp_dir"].iloc[0]

    # TODO sample_id not defined?  barcodes uniquely identify each spot per slice,
    #      aggregated across slices/bams.
    df_agg_barcode = get_aggregated_barcodes(f"{snp_dir}/barcodes.txt")

    assert (alignment_files is None) or (
        len(alignment_files) + 1 == df_meta.shape[0]
    ), "TODO!"

    # TODO duplicate of df_agg_barcode
    snp_barcodes = pd.read_csv(
        f"{snp_dir}/barcodes.txt", header=None, names=["barcodes"]
    )

    unique_snp_ids = np.load(f"{snp_dir}/unique_snp_ids.npy", allow_pickle=True)

    ##### read SNP counts #####
    cell_snp_Aallele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Aallele.npz")
    cell_snp_Ballele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Ballele.npz")

    ##### read anndata and coordinate #####
    adata = None

    # NB df_meta provides the sample_ids, one per bam.
    for i, sname in enumerate(df_meta.sample_id.values):
        index = np.where(df_agg_barcode["sample_id"] == sname)[0]

        df_this_barcode = copy.copy(df_agg_barcode.iloc[index, :])
        df_this_barcode.index = df_this_barcode.barcode

        # NB read filtered_feature_bc_matrix.h5(ad) from spaceranger_dir for
        #    for this sample.
        adatatmp = get_spaceranger_counts(df_meta["spaceranger_dir"].iloc[i])

        # NB reorder anndata spots to have the order of df_this_barcode (with enum)
        idx_argsort = pd.Categorical(
            adatatmp.obs.index, categories=list(df_this_barcode.barcode), ordered=True
        ).argsort()

        # TODO UGH
        adatatmp = adatatmp[idx_argsort, :]

        # NB limited to in tissue by default.
        df_this_pos = get_spatial_positions(df_meta["spaceranger_dir"].iloc[i])

        # NB only keep shared barcodes between (IN_TISSUE) visium barcodes and filtered_feature_bc_matrix.
        shared_barcodes = set(list(df_this_pos.barcode)) & set(list(adatatmp.obs.index))

        isin = adatatmp.obs.index.isin(shared_barcodes)

        logger.info(
            f"Retaining {100. * np.mean(isin):.3f}% of barcodes (shared with bam & filtered matrix) for {sname}."
        )

        # TODO filter before sort.
        adatatmp = adatatmp[isin, :]

        df_this_pos = df_this_pos[df_this_pos.barcode.isin(shared_barcodes)]

        # NB re-order positions to have order of df_this_barcode barcodes.
        df_this_pos.barcode = pd.Categorical(
            df_this_pos.barcode, categories=list(adatatmp.obs.index), ordered=True
        )

        df_this_pos.sort_values(by="barcode", inplace=True)

        adatatmp.obsm["X_pos"] = np.vstack([df_this_pos.x, df_this_pos.y]).T
        adatatmp.obs["sample"] = sname

        # NB index by {barcode}_{sample} (TBC)
        adatatmp.obs.index = [f"{x}_{sname}" for x in adatatmp.obs.index]

        adatatmp.var_names_make_unique()

        if adata is None:
            adata = adatatmp
        else:
            adata = anndata.concat([adata, adatatmp], join="outer")

    # NB replace nan with 0 and cast to int.
    adata.layers["count"][np.isnan(adata.layers["count"])] = 0
    adata.layers["count"] = adata.layers["count"].astype(int)

    ##### filter by spots #####
    
    # NB shared barcodes between adata and SNPs.
    shared_barcodes = set(list(snp_barcodes.barcodes)) & set(list(adata.obs.index))

    isin = snp_barcodes.barcodes.isin(shared_barcodes)

    logger.info(
        f"Retaining {100. * np.mean(isin):.3f}% of SNP barcodes (shared between UMIs and SNPs)."
    )

    cell_snp_Aallele = cell_snp_Aallele[isin, :]
    cell_snp_Ballele = cell_snp_Ballele[isin, :]

    snp_barcodes = snp_barcodes[isin]

    isin = adata.obs.index.isin(shared_barcodes)

    logger.info(
        f"Retaining {100. * np.mean(isin):.3f}% of UMI barcodes (shared between UMIs and SNPs)."
    )

    adata = adata[isin, :]
    adata = adata[
        pd.Categorical(
            adata.obs.index, categories=list(snp_barcodes.barcodes), ordered=True
        ).argsort(),
        :,
    ]

    across_slice_adjacency_mat = get_alignments(alignment_files, df_meta)

    # NB filter out spots with too small number of UMIs;
    # TODO differentiate min_snpumis; why before genomic binning?
    indicator = np.sum(adata.layers["count"], axis=1) >= min_snpumis

    logger.info(f"Retaining {100. * np.mean(isin):.3f}% of spots with sufficient UMIs")

    adata = adata[indicator, :]

    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]
    
    if across_slice_adjacency_mat is not None:
        across_slice_adjacency_mat = across_slice_adjacency_mat[indicator, :][
            :, indicator
        ]

    # NB filter out spots with too small number of snp-covering UMIs;
    # TODO indicator &= indicator ...
    indicator = (
        np.sum(cell_snp_Aallele, axis=1).A.flatten()
        + np.sum(cell_snp_Ballele, axis=1).A.flatten()
        >= min_snpumis
    )

    adata = adata[indicator, :]

    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]
    
    if not (across_slice_adjacency_mat is None):
        across_slice_adjacency_mat = across_slice_adjacency_mat[indicator, :][
            :, indicator
        ]

    ##### filter by ranges #####
        
    # NB filter out genes that are expressed in <min_percent_expressed_spots cells
    # TODO apply @ get_spaceranger_counts
    indicator = (
        np.sum(adata.X > 0, axis=0) >= min_percent_expressed_spots * adata.shape[0]
    ).A.flatten()

    logger.info(
        f"Retaining {100. * np.mean(isin):.3f}% of genes with sufficient expression across spots"
    )

    # DEPRECATE
    # genenames = set(list(adata.var.index[indicator]))
    adata = adata[:, indicator]

    logger.info(
        f"median UMI after filtering out genes < {100. * min_percent_expressed_spots}% of cells = {np.median(np.sum(adata.layers["count"], axis=1))}"
    )
    
    if filtergenelist_file is not None:
        filter_gene_list = get_filtergenelist(filtergenelist_file)

        # TODO slow.
        indicator_filter = np.array(
            [(not x in filter_gene_list) for x in adata.var.index]
        )

        adata = adata[:, indicator_filter]

        logger.info(
            f"median UMI after filtering out genes in {filtergenelist_file} = {np.median(np.sum(adata.layers["count"], axis=1))}"
        )

        # TODO?
        # apply ranges cut to cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids?

    if filter_range_file is not None:
        ranges = get_filter_ranges(filter_range_file)

        indicator_filter = np.array([True] * cell_snp_Aallele.shape[1])
        j = 0

        # TODO read-through / slow.
        for i in range(cell_snp_Aallele.shape[1]):
            this_chr = int(unique_snp_ids[i].split("_")[0])
            this_pos = int(unique_snp_ids[i].split("_")[1])

            # NB fast forward genomic position
            while j < ranges.shape[0] and (
                (ranges.Chr.values[j] < this_chr)
                or (
                    (ranges.Chr.values[j] == this_chr)
                    and (ranges.End.values[j] <= this_pos)
                )
            ):
                j += 1
                
            if (
                j < ranges.shape[0]
                and (ranges.Chr.values[j] == this_chr)
                and (ranges.Start.values[j] <= this_pos)
                and (ranges.End.values[j] > this_pos)
            ):
                indicator_filter[i] = False

        cell_snp_Aallele = cell_snp_Aallele[:, indicator_filter]
        cell_snp_Ballele = cell_snp_Ballele[:, indicator_filter]

        unique_snp_ids = unique_snp_ids[indicator_filter]

    if local_outlier_filter:
        # NB  k-NN defined density estimates used to filter local outliers given density wrt neighbors.
        #     see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
        #         https://en.wikipedia.org/wiki/Local_outlier_factor
        clf = LocalOutlierFactor(n_neighbors=200)

        # NB  prediction on spot/barcode summed transcripts for each gene.
        label = clf.fit_predict(np.sum(adata.layers["count"], axis=0).reshape(-1, 1))

        # TODO?  zeros out counts.
        adata.layers["count"][:, np.where(label == -1)[0]] = 0

        logger.info("filter out {} outlier genes.".format(np.sum(label == -1)))

    if normalidx_file is not None:
        normal_barcodes = pd.read_csv(normalidx_file, header=None).iloc[:, 0].values
        adata.obs["tumor_annotation"] = "tumor"
        adata.obs["tumor_annotation"][adata.obs.index.isin(normal_barcodes)] = "normal"

        logger.info("Applied tumor annotation: {adata.obs['tumor_annotation'].value_counts()}")

    logger.info("Realized AnnData:\n{adata}")
        
    return (
        adata,
        cell_snp_Aallele.A,
        cell_snp_Ballele.A,
        unique_snp_ids,
        across_slice_adjacency_mat,
    )
