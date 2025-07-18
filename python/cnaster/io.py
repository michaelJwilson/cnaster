import copy
import logging
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from cnaster.filter import get_filter_genes, get_filter_ranges
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)


def get_sample_sheet(sample_sheet_path):
    df_meta = pd.read_csv(sample_sheet_path, sep="\t")

    logger.info(f"Input sample_sheet_path={sample_sheet_path} contains:\n{df_meta}")

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

    # TODO sample ids currently slice, e.g. U1;
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

        logger.info(f"Reading {spaceranger_dir}/spatial/tissue_positions.csv")

    elif Path(f"{spaceranger_dir}/spatial/tissue_positions_list.csv").exists():
        df_this_pos = pd.read_csv(
            f"{spaceranger_dir}/spatial/tissue_positions_list.csv",
            sep=",",
            header=None,
            names=names,
        )

        logger.info(f"Reading {spaceranger_dir}/spatial/tissue_positions_list.csv")

    else:
        logger.error(f"No spatial coordinate file @ {spaceranger_dir}.")
        raise RuntimeError()

    # TODO alignment defined for in_tissue == True only?
    if filter_in_tissue:
        result = df_this_pos[df_this_pos.in_tissue == True]
    else:
        result = df_this_pos

    return result


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
    adatatmp.layers["count"] = adatatmp.X.toarray()

    is_nan = np.isnan(adatatmp.layers["count"])

    logger.info(
        f"Found {100.0 * np.mean(is_nan):.3f}% NaN counts in anndata.  zeroing."
    )

    # NB replace nan with 0 and cast to int.
    adatatmp.layers["count"][is_nan] = 0
    adatatmp.layers["count"] = adatatmp.layers["count"].astype(int)

    # e.g. duplicated:  TBCE  2, LINC01238  2.3; why?
    # duplicated_mask = adatatmp.var_names.duplicated(keep=False)
    # non_unique_vars = adatatmp.var_names[duplicated_mask]

    # duplicate_counts = non_unique_vars.value_counts()

    # NB var names made unique by appending an index string,
    #    see https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.var_names_make_unique.html
    adatatmp.var_names_make_unique()

    logger.info(f"Read features of shape {adatatmp.shape} from {spaceranger_dir}")

    # NB data matrix X (ndarray/csr matrix, dask ...): observations/cells are named by their barcode and variables/genes by gene name
    return adatatmp


# TODO massively inefficient?
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

    # TODO HACK assumes snps derived from aggregation of all provided samples.
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

    # TODO HACK
    sample_id_patcher = {
        sample_id.split("-")[1]: sample_id for sample_id in df_meta.sample_id.values
    }

    df_agg_barcode["sample_id"] = df_agg_barcode["sample_id"].map(sample_id_patcher)

    # TODO HACK -/_
    snp_barcodes["barcodes"] = snp_barcodes["barcodes"].map(
        lambda xx: xx.split("_")[0] + "_" + sample_id_patcher[xx.split("_")[-1]]
    )

    unique_snp_ids = np.load(f"{snp_dir}/unique_snp_ids.npy", allow_pickle=True)

    ##### read SNP counts #####
    cell_snp_Aallele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Aallele.npz")
    cell_snp_Ballele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Ballele.npz")

    ##### read anndata and coordinate #####
    adata = None

    # NB df_meta provides the sample_ids, one per bam.
    for i, sname in enumerate(df_meta.sample_id.values):
        logger.info(f"Solving for spaceranger sample {sname}.")

        # NB barcodes for this sample + slice.
        index = np.where(df_agg_barcode["sample_id"] == sname)[0]

        df_this_barcode = copy.copy(df_agg_barcode.iloc[index, :])
        df_this_barcode.index = df_this_barcode.barcode

        # NB limited to in tissue by default.
        df_this_pos = get_spatial_positions(df_meta["spaceranger_dir"].iloc[i])

        # NB read filtered_feature_bc_matrix.h5(ad) from spaceranger_dir for
        #    for this sample.
        adatatmp = get_spaceranger_counts(df_meta["spaceranger_dir"].iloc[i])

        # NB reorder anndata spots to have the order of df_this_barcode (with enum)
        idx_argsort = pd.Categorical(
            adatatmp.obs.index, categories=list(df_this_barcode.barcode), ordered=True
        ).argsort()

        # TODO UGH
        adatatmp = adatatmp[idx_argsort, :].copy()

        # NB only keep shared barcodes between (IN_TISSUE) visium barcodes and filtered_feature_bc_matrix.
        shared_barcodes = set(list(df_this_pos.barcode)) & set(list(adatatmp.obs.index))

        isin = adatatmp.obs.index.isin(shared_barcodes)

        logger.info(
            f"Retaining {100.0 * np.mean(isin):.3f}% of barcodes (shared with bam & filtered matrix) for {sname}."
        )

        # TODO filter before sort.
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
        adatatmp.obs.index = [f"{x}_{sname}" for x in adatatmp.obs.index]

        if adata is None:
            adata = adatatmp
        else:
            adata = anndata.concat([adata, adatatmp], join="outer")

    ##### filter by spots #####

    # NB shared barcodes between adata and SNPs.
    shared_barcodes = set(list(snp_barcodes.barcodes)) & set(list(adata.obs.index))

    isin = snp_barcodes.barcodes.isin(shared_barcodes).values

    logger.info(
        f"Retaining {100.0 * np.mean(isin):.3f}% of SNP barcodes (shared between UMIs and SNPs)."
    )

    cell_snp_Aallele = cell_snp_Aallele[isin, :]
    cell_snp_Ballele = cell_snp_Ballele[isin, :]

    snp_barcodes = snp_barcodes[isin]

    isin = adata.obs.index.isin(shared_barcodes)

    logger.info(
        f"Retaining {100.0 * np.mean(isin):.3f}% of UMI barcodes (shared between UMIs and SNPs)."
    )

    adata = adata[isin, :].copy()
    adata = adata[
        pd.Categorical(
            adata.obs.index, categories=list(snp_barcodes.barcodes), ordered=True
        ).argsort(),
        :,
    ]

    across_slice_adjacency_mat = get_alignments(
        alignment_files, df_meta, df_agg_barcode
    )

    # NB filter out spots with too small number of UMIs;
    # TODO differentiate min_snpumis; why before genomic binning?
    indicator = np.sum(adata.layers["count"], axis=1) >= min_snp_umis

    logger.info(
        f"Retaining {100.0 * np.mean(indicator):.3f}% of spots with sufficient UMIs"
    )

    indicator &= (
        np.sum(cell_snp_Aallele, axis=1).A.flatten()
        + np.sum(cell_snp_Ballele, axis=1).A.flatten()
        >= min_snp_umis
    )

    logger.info(
        f"Retaining {100.0 * np.mean(indicator):.3f}% of spots with sufficient snp UMIs"
    )

    adata = adata[indicator, :]

    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]

    if across_slice_adjacency_mat is not None:
        across_slice_adjacency_mat = across_slice_adjacency_mat[indicator, :][
            :, indicator
        ]

    # NB filter out genes that are expressed in <min_percent_expressed_spots cells
    # TODO apply @ get_spaceranger_counts
    indicator = (
        np.sum(adata.X > 0, axis=0) >= min_percent_expressed_spots * adata.shape[0]
    ).A.flatten()

    # NB total UMIs for all spots given (selected) genes.
    ratio = np.sum(adata.X[:, indicator]) / np.sum(adata.X)

    # TODO gencode gene list is not all sampled by (3') visium umis.
    # TODO excludes 50% of genes, but retains 99.97% of UMIs; resolves gene definition to house-keeping?
    logger.info(
        f"Retaining {100.0 * np.mean(indicator):.3f}% of genes with sufficient expression across spots ({100.0 * ratio:.2f}% of total UMIs)."
    )

    adata = adata[:, indicator]

    logger.info(
        f"Median UMI after gene selection for expression < {100.0 * min_percent_expressed_spots:.3f}% of cells = {np.median(np.sum(adata.layers['count'], axis=1))}"
    )

    if filter_gene_file is not None:
        genes_to_filter = get_filter_genes(filter_gene_file).iloc[:, 0].values
        indicator_filter = ~np.isin(adata.var.index, genes_to_filter)

        logger.info(
            f"Removing genes based on input ranges ({filter_gene_file}):"
        )

        for to_print in genes_to_filter[np.isin(genes_to_filter, adata.var.index)]:
            print(to_print) 

        adata = adata[:, indicator_filter]

        logger.info(
            f"Median UMI after filtering genes = {np.median(np.sum(adata.layers['count'], axis=1))}"
        )

        # TODO?
        # apply ranges cut to cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids?

    if filter_range_file is not None:
        ranges = get_filter_ranges(filter_range_file)
        num_ranges = ranges.shape[0]

        indicator_filter = np.array([True] * cell_snp_Aallele.shape[1])
        j = 0

        # TODO read-through / slow.
        for i in range(cell_snp_Aallele.shape[1]):
            this_chr = int(unique_snp_ids[i].split("_")[0])
            this_pos = int(unique_snp_ids[i].split("_")[1])

            # NB fast forward genomic position
            while j < num_ranges and (
                (ranges.Chr.values[j] < this_chr)
                or (
                    (ranges.Chr.values[j] == this_chr)
                    and (ranges.End.values[j] <= this_pos)
                )
            ):
                j += 1

            if (
                j < num_ranges
                and (ranges.Chr.values[j] == this_chr)
                and (ranges.Start.values[j] <= this_pos)
                and (ranges.End.values[j] > this_pos)
            ):
                indicator_filter[i] = False

        logger.info(
            f"Retaining {100.0 * np.mean(indicator_filter):.2f}% of SNPs based on input filter ranges."
        )

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

        to_zero = np.where(label == -1)[0]
        ratio = np.sum(adata.layers["count"][:, to_zero]) / np.sum(
            adata.layers["count"]
        )

        # TODO removed 235 outlier genes (51.310% of UMIs)!!
        logger.info(
            f"Removed {len(to_zero)} outlier genes ({100.0 * ratio:.3f}% of UMIs) based on {clf.__class__.__name__}."
        )

        if len(to_zero) > 0:
            gene_umi_counts = np.sum(adata.layers["count"], axis=0)
            total_umis = np.sum(adata.layers["count"])
            
            outlier_genes_info = []

            for gene_idx in to_zero:
                gene_name = adata.var.index[gene_idx]
                gene_umis = gene_umi_counts[gene_idx]
                gene_pct = 100.0 * gene_umis / total_umis
                outlier_genes_info.append((gene_name, gene_umis, gene_pct))
            
            outlier_genes_info.sort(key=lambda x: x[2], reverse=True)
            
            logger.info("Top 10 outlier genes removed:")

            for i, (gene_name, gene_umis, gene_pct) in enumerate(outlier_genes_info[:10]):
                logger.info(f"  {i+1}. {gene_name}: {gene_pct:.3f}% UMIs")

        # NB zero count of outlier genes.
        adata.layers["count"][:, to_zero] = 0
    if normal_idx_file is not None:
        normal_barcodes = pd.read_csv(normalidx_file, header=None).iloc[:, 0].values
        adata.obs["tumor_annotation"] = "tumor"
        adata.obs["tumor_annotation"][adata.obs.index.isin(normal_barcodes)] = "normal"

        logger.info(
            "Applied tumor annotation: {adata.obs['tumor_annotation'].value_counts()}"
        )

    logger.info(f"Realized AnnData:\n{adata}")

    # TODO dense arrays.
    return (
        adata,
        cell_snp_Aallele.toarray(),
        cell_snp_Ballele.toarray(),
        unique_snp_ids,
        across_slice_adjacency_mat,
    )
