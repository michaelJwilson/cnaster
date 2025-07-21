import logging
import scipy
import scipy.stats
import anndata
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans

from cnaster.hmm_emission import Weighted_BetaBinom
from cnaster.reference import get_reference_recomb_rates
from cnaster.recomb import assign_centiMorgans, compute_numbat_phase_switch_prob


logger = logging.getLogger(__name__)


def binned_gene_snp(df_gene_snp):
    # NB table with contig range + set of genes + snp_ids.
    table_bininfo = (
        df_gene_snp[~df_gene_snp.bin_id.isnull()]
        .groupby("bin_id")
        .agg(
            {
                "CHR": "first",
                "START": "first",
                "END": "last",
                "gene": set,
                "snp_id": set,
            }
        )
        .reset_index()
    )
    table_bininfo["ARM"] = "."
    table_bininfo["INCLUDED_GENES"] = [
        " ".join([x for x in y if not x is None]) for y in table_bininfo.gene.values
    ]
    table_bininfo["INCLUDED_SNP_IDS"] = [
        " ".join([x for x in y if not x is None]) for y in table_bininfo.snp_id.values
    ]
    table_bininfo["NORMAL_COUNT"] = np.nan
    table_bininfo["N_SNPS"] = [
        len([x for x in y if not x is None]) for y in table_bininfo.snp_id.values
    ]

    table_bininfo.drop(columns=["gene", "snp_id"], inplace=True)
    return table_bininfo


def filter_normal_diffexp(
    exp_counts,
    df_bininfo,
    normal_candidate,
    sample_list=None,
    sample_ids=None,
    logfcthreshold_u=2,
    logfcthreshold_t=4,
    quantile_threshold=80,
):
    """
    Attributes
    ----------
    df_bininfo : pd.DataFrame
        Contains columns ['CHR', 'START', 'END', 'INCLUDED_GENES', 'INCLUDED_SNP_IDS'], 'INCLUDED_GENES' contains space-delimited gene names.
    """
    adata = anndata.AnnData(exp_counts)
    adata.layers["count"] = exp_counts.values
    adata.obs["normal_candidate"] = normal_candidate

    map_gene_adatavar, map_gene_umi = {}, {}
    list_gene_umi = np.sum(adata.layers["count"], axis=0)

    for i, x in enumerate(adata.var.index):
        map_gene_adatavar[x] = i
        map_gene_umi[x] = list_gene_umi[i]

    if sample_list is None:
        sample_list = [None]

    filtered_out_set = set()

    for s, sname in enumerate(sample_list):
        if sname is None:
            index = np.arange(adata.shape[0])
        else:
            index = np.where(sample_ids == s)[0]
        tmpadata = adata[index, :].copy()
        if (
            np.sum(tmpadata.layers["count"][tmpadata.obs["normal_candidate"], :])
            < tmpadata.shape[1] * 10
        ):
            continue

        umi_threshold = np.percentile(
            np.sum(tmpadata.layers["count"], axis=0), quantile_threshold
        )

        sc.pp.filter_genes(tmpadata, min_cells=10)
        med = np.median(np.sum(tmpadata.layers["count"], axis=1))

        sc.pp.normalize_total(tmpadata, target_sum=med)
        sc.pp.log1p(tmpadata)

        sc.pp.pca(tmpadata, n_comps=4)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(tmpadata.obsm["X_pca"])
        kmeans_labels = kmeans.predict(tmpadata.obsm["X_pca"])
        idx_kmeans_label = np.argmax(
            np.bincount(kmeans_labels[tmpadata.obs["normal_candidate"]], minlength=2)
        )
        clone = np.array(["normal"] * tmpadata.shape[0])
        clone[
            (kmeans_labels != idx_kmeans_label) & (~tmpadata.obs["normal_candidate"])
        ] = "tumor"

        clone[
            (kmeans_labels == idx_kmeans_label) & (~tmpadata.obs["normal_candidate"])
        ] = "unsure"
        tmpadata.obs["clone"] = clone

        agg_counts = np.vstack(
            [
                np.sum(tmpadata.layers["count"][tmpadata.obs["clone"] == c, :], axis=0)
                for c in ["normal", "unsure", "tumor"]
            ]
        )
        agg_counts = agg_counts / np.sum(agg_counts, axis=1, keepdims=True) * 1e6
        geneumis = np.array([map_gene_umi[x] for x in tmpadata.var.index])
        logfc_u = np.where(
            ((agg_counts[1, :] == 0) | (agg_counts[0, :] == 0)),
            10,
            np.log2(agg_counts[1, :] / agg_counts[0, :]),
        )
        logfc_t = np.where(
            ((agg_counts[2, :] == 0) | (agg_counts[0, :] == 0)),
            10,
            np.log2(agg_counts[2, :] / agg_counts[0, :]),
        )
        this_filtered_out_set = set(
            list(
                tmpadata.var.index[
                    (np.abs(logfc_u) > logfcthreshold_u) & (geneumis > umi_threshold)
                ]
            )
        ) | set(
            list(
                tmpadata.var.index[
                    (np.abs(logfc_t) > logfcthreshold_t) & (geneumis > umi_threshold)
                ]
            )
        )
        filtered_out_set = filtered_out_set | this_filtered_out_set
        print(f"Filter out {len(filtered_out_set)} DE genes")

    new_single_X_rdr = np.zeros((df_bininfo.shape[0], adata.shape[0]))

    for b, genestr in enumerate(df_bininfo.INCLUDED_GENES.values):
        # RDR (genes)
        involved_genes = set(genestr.split(" ")) - filtered_out_set
        new_single_X_rdr[b, :] = np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)], axis=1
        )

    return new_single_X_rdr, filtered_out_set


def normal_baf_bin_filter(
    df_gene_snp,
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    nu,
    logphase_shift,
    index_normal,
    geneticmap_file,
    confidence_interval=(0.05, 0.95),
    min_betabinom_tau=30,
):
    """
    Remove bins that potentially contain somatic mutations based on normal spot BAF.
    """
    logger.info("Selecting bins for removal based on normal spot BAF.")

    # NB pool b-allele counts for each bin across all normal spots
    tmpX = np.sum(single_X[:, 1, index_normal], axis=1)
    tmptotal_bb_RD = np.sum(single_total_bb_RD[:, index_normal], axis=1)
    model = Weighted_BetaBinom(
        tmpX, np.ones(len(tmpX)), weights=np.ones(len(tmpX)), exposure=tmptotal_bb_RD
    )
    tmpres = model.fit(disp=0)
    tmpres.params[0] = 0.5
    tmpres.params[-1] = max(tmpres.params[-1], min_betabinom_tau)

    # NB remove bins if "normal" b-allele frequencies fall out of 5%-95% probability range
    removal_indicator1 = tmpX < scipy.stats.betabinom.ppf(
        confidence_interval[0],
        tmptotal_bb_RD,
        tmpres.params[0] * tmpres.params[1],
        (1.0 - tmpres.params[0]) * tmpres.params[1],
    )
    removal_indicator2 = tmpX > scipy.stats.betabinom.ppf(
        confidence_interval[1],
        tmptotal_bb_RD,
        tmpres.params[0] * tmpres.params[1],
        (1.0 - tmpres.params[0]) * tmpres.params[1],
    )

    index_removal = np.where(removal_indicator1 | removal_indicator2)[0]
    index_remaining = np.where(~(removal_indicator1 | removal_indicator2))[0]

    col = np.where(df_gene_snp.columns == "bin_id")[0][0]
    df_gene_snp.iloc[np.where(df_gene_snp.bin_id.isin(index_removal))[0], col] = None

    df_gene_snp["bin_id"] = df_gene_snp["bin_id"].map(
        {x: i for i, x in enumerate(index_remaining)}
    )
    df_gene_snp.bin_id = df_gene_snp.bin_id.astype("Int64")

    single_X = single_X[index_remaining, :, :]
    single_base_nb_mean = single_base_nb_mean[index_remaining, :]
    single_total_bb_RD = single_total_bb_RD[index_remaining, :]

    lengths = np.zeros(len(df_gene_snp.CHR.unique()), dtype=int)

    for i, c in enumerate(df_gene_snp.CHR.unique()):
        lengths[i] = len(
            df_gene_snp[
                (df_gene_snp.CHR == c) & (~df_gene_snp.bin_id.isnull())
            ].bin_id.unique()
        )

    # NB phase switch probability from genetic distance
    sorted_chr_pos_first = df_gene_snp.groupby("bin_id").agg(
        {"CHR": "first", "START": "first"}
    )
    sorted_chr_pos_first = list(
        zip(sorted_chr_pos_first.CHR.values, sorted_chr_pos_first.START.values)
    )
    sorted_chr_pos_last = df_gene_snp.groupby("bin_id").agg(
        {"CHR": "last", "END": "last"}
    )
    sorted_chr_pos_last = list(
        zip(sorted_chr_pos_last.CHR.values, sorted_chr_pos_last.END.values)
    )

    tmp_sorted_chr_pos = [
        val for pair in zip(sorted_chr_pos_first, sorted_chr_pos_last) for val in pair
    ]

    ref_positions_cM = get_reference_recomb_rates(geneticmap_file)

    position_cM = assign_centiMorgans(tmp_sorted_chr_pos, ref_positions_cM)

    phase_switch_prob = compute_numbat_phase_switch_prob(
        position_cM, tmp_sorted_chr_pos, nu
    )

    log_sitewise_transmat = np.minimum(
        np.log(0.5), np.log(phase_switch_prob) - logphase_shift
    )

    log_sitewise_transmat = log_sitewise_transmat[
        np.arange(1, len(log_sitewise_transmat), 2)
    ]

    return (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        log_sitewise_transmat,
        df_gene_snp,
    )
