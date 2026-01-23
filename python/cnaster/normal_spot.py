import ast
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
from cnaster.hmm_utils import get_em_solver_params
from cnaster.config import get_global_config
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


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


def determine_normal_candidates(
    config,
    res,
    baf_profiles,
    single_X,
    single_X_rdr,
    smooth_mat,
    single_tumor_prop=None,
):
    """
    Determine normal candidate spots based on BAF profiles, tumor proportion, or provided indices.
    Returns a boolean array normal_candidate.
    """
    logger.info(f"Determining normal spots based on BAF-only clones.")

    # NB no input files for barcodes of normal spots, or tumor proportion per spot.
    if (config.preprocessing.normalidx_file is None) and (
        config.preprocessing.tumorprop_file is None
    ):
        EPS_BAF = 0.05 # MAGIC
        PERCENT_NORMAL = 40 # MAGIC

        logger.info(
            f"Identifying normal spots based on estimated BAF given EPS_BAF={EPS_BAF} and PERCENT_NORMAL={PERCENT_NORMAL}."
        )

        # NB sum deviations > EPS_BAF from 0.5 along the genome for each clone; pick normal as minimum deviation.
        baf_deviations = np.sum(
            np.maximum(np.abs(baf_profiles - 0.5) - EPS_BAF, 0), axis=1
        )
        id_nearnormal_clone = np.argmin(baf_deviations)

        logger.info(
            f"Found clone {id_nearnormal_clone} to be the most normal-like given BAF deviations."
        )

        # NB measure the standard deviation of log-transformed, smoothed transcript counts for each spot.
        vec_stds = np.std(np.log1p(single_X_rdr @ smooth_mat), axis=0)
        prior_stdthreshold = np.inf

        while True:
            stdthreshold = np.percentile(
                vec_stds[res["new_assignment"] == id_nearnormal_clone],
                PERCENT_NORMAL,
            )
            normal_candidate = (vec_stds < stdthreshold) & (
                res["new_assignment"] == id_nearnormal_clone
            )

            if config.run.legacy and (
                np.sum(single_X_rdr[:, (normal_candidate == True)])
                > 200 * single_X.shape[0]  # MAGIC
            ):
                logger.info(
                    f"Assumed legacy normal spot allocation for {PERCENT_NORMAL}[%] normal spots"
                )
                break

            elif stdthreshold > 1.5 * prior_stdthreshold: # MAGIC
                logger.info(
                    f"Determined {PERCENT_NORMAL}% normal spots with sufficient UMIs, assigned to normal like clone."
                )
                logger.info(
                    f"BAF-clone breakdown:\n{np.unique(res['new_assignment'][normal_candidate], return_counts=True)}"
                )
                break
            elif PERCENT_NORMAL == 100:
                logger.warning(
                    f"All {np.count_nonzero(res['new_assignment'] == id_nearnormal_clone)} spots for clone {id_nearnormal_clone} considered to be normal."
                )
                break

            PERCENT_NORMAL += 10
        return normal_candidate

    elif config.preprocessing.normalidx_file is not None:
        # single_base_nb_mean has already been added in loading data step.
        if config.preprocessing.tumorprop_file is not None:
            logger.warning(
                f"Found mixed sources for normal spot definition, assuming {config.preprocessing.normalidx_file}."
            )
        # You may want to load normal_candidate from file here
        return None

    else:
        assert single_tumor_prop is not None

        logger.info(f"Identifying normal spots based on provided tumor proportion.")

        for prop_threshold in np.arange(0.05, 0.6, 0.05):
            # NB suggests 0 is perfectly normal and otherwise measures tumor proportion, sensibly!
            normal_candidate = single_tumor_prop < prop_threshold

            if (
                np.sum(single_X_rdr[:, (normal_candidate == True)])
                > 200 * single_X.shape[0]  # MAGIC
            ):
                logger.info(
                    f"Determined normal spots with sufficient UMIs based on input tumor proportion @ prop_threshold={prop_threshold}"
                )
                break
        else:
            logger.warning(
                f"Failed to determine normal spots with sufficient UMIs based on input tumor proportion."
            )
        return normal_candidate


def determine_normal_baseline(single_X_rdr, normal_candidate, config=None):
    if config is None:
        config = get_global_config()

    logger.info(
        f"Found sparsity of normal spot set={100. * np.mean(single_X_rdr[:, (normal_candidate == True)]) == 0.:.3f}%"
    )

    # NB normal baseline transcript count; unnormalized.
    # TODO normal_candidate clause
    rdr_normal = np.sum(single_X_rdr[:, (normal_candidate == True)], axis=1)

    bidx_inconfident = np.where(rdr_normal < config.quality.min_normal_count_perbin)[0]

    logger.info(
        f"Found {100. * np.mean(rdr_normal >= config.quality.min_normal_count_perbin):.3f}% of segments with confident normal baseline for MIN_NORMAL_COUNT_PERBIN={config.quality.min_normal_count_perbin}"
    )

    # NB where normal transcript count < config.quality.min_normal_count_perbin, zero.
    rdr_normal[bidx_inconfident] = 0
    rdr_normal = rdr_normal / np.sum(rdr_normal)

    # NB avoid ill-defined distributions if normal has 0 count in that bin.
    single_X_rdr[bidx_inconfident, :] = 0

    # NB replicate and normalize rdr_normal to the per-spot total transcripts, T_n.
    spots_coverage = np.sum(single_X_rdr, axis=0)

    single_base_nb_mean = rdr_normal.reshape(-1, 1) @ spots_coverage.reshape(1, -1)

    return rdr_normal, single_X_rdr, single_base_nb_mean


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
    Identify and filter out genes that are differentially expressed between "normal" candidates & other cell populations (such as tumor cells)
    in a dataset based on statistical tests.

    Attributes
    ----------
    df_bininfo : pd.DataFrame
        Contains columns ['CHR', 'START', 'END', 'INCLUDED_GENES', 'INCLUDED_SNP_IDS'], 'INCLUDED_GENES' contains space-delimited gene names.
    """
    adata = anndata.AnnData(exp_counts)
    adata.layers["count"] = exp_counts.values
    adata.obs["normal_candidate"] = normal_candidate

    map_gene_adatavar, map_gene_umi = {}, {}

    # NB gene_umis summed over spots.
    list_gene_umi = np.sum(adata.layers["count"], axis=0)

    # NB map of unique integer per gene.
    for i, x in enumerate(adata.var.index):
        map_gene_adatavar[x] = i
        map_gene_umi[x] = list_gene_umi[i]

    if sample_list is None:
        sample_list = [None]

    filtered_out_set = set()

    # NB loop over slices.
    for s, sname in enumerate(sample_list):
        if sname is None:
            index = np.arange(adata.shape[0])
        else:
            index = np.where(sample_ids == s)[0]
        tmpadata = adata[index, :].copy()
        if (
            np.sum(tmpadata.layers["count"][tmpadata.obs["normal_candidate"], :])
            < tmpadata.shape[1] * 10  # MAGIC
        ):
            logger.warning(f"TODO!")
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

        # TODO divide-by-zero errors >>>>
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
        # <<<<<
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

        logger.info(
            f"Removed {len(filtered_out_set)} genes with differential expression based on normal spots."
        )

    new_single_X_rdr = np.zeros((df_bininfo.shape[0], adata.shape[0]))
    total_counts, retained_counts = 0, 0

    for b, genestr in enumerate(df_bininfo.INCLUDED_GENES.values):
        # RDR (genes)
        bin_genes = set(genestr.split(" "))
        involved_genes = bin_genes - filtered_out_set

        total_counts += np.sum(
            adata.layers["count"][:, adata.var.index.isin(bin_genes)]
        )
        retained_counts += np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)]
        )

        new_single_X_rdr[b, :] = np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)], axis=1
        )

    logger.info(f"Retained {100. * retained_counts / total_counts:.3f}% of bin UMIs.")

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
    confidence_interval=None,
    min_betabinom_tau=30,
):
    """
    Calculate new (block, spot) counts after filtering genomic bins that have non-normal-like 
    baf.  This may be the case if mixed with non-normal spots or allele-specific expression.
    """
    if confidence_interval is None:
        confidence_interval = ast.literal_eval(
            get_global_config().quality.normal_allele_specific_confidence
        )

    logger.info("Selecting bins for removal based on normal spot BAF.")

    # NB pool b-allele counts for each bin across all normal spots; 1D genomic segments.
    tmpX = np.sum(single_X[:, 1, index_normal], axis=1)
    tmptotal_bb_RD = np.sum(single_total_bb_RD[:, index_normal], axis=1)

    # TODO
    model = Weighted_BetaBinom(
        tmpX, np.ones(len(tmpX)), weights=np.ones(len(tmpX)), exposure=tmptotal_bb_RD
    )

    # LEGACY
    settings = get_em_solver_params()
    tmpres = model.fit(**settings)

    logger.info(
        f"Best-fit BetaBinom model to normal spot BAF has parameters={tmpres.params}"
    )

    # TODO warn if patched.
    # NB patches parameters assuming min_betabinom_tau=30;
    tmpres.params[0] = 0.5
    tmpres.params[-1] = max(tmpres.params[-1], min_betabinom_tau)

    # NB remove bins if "normal" b-allele probabilities fall out of (5%-95%) confidence interval,
    #    this may be the case if mixed with non-normal spots or allele-specific expression present.
    #    
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

    logger.info(
        f"Removing {100. * np.mean(removal_indicator1 | removal_indicator2):.4f}% of genomic bins with potential allele-specific expression based on normal spot candidates assuming confidence={confidence_interval} and min_betabinom_tau={min_betabinom_tau}."
    )

    # NB below constructs single_X, single_base_nb_mean, single_total_bb_RD with segments removed.
    col = np.where(df_gene_snp.columns == "bin_id")[0][0]
    df_gene_snp.iloc[np.where(df_gene_snp.bin_id.isin(index_removal))[0], col] = None

    df_gene_snp["bin_id"] = df_gene_snp["bin_id"].map(
        {x: i for i, x in enumerate(index_remaining)}
    )
    df_gene_snp.bin_id = df_gene_snp.bin_id.astype("Int64")

    logger.info(f"Solved for unique bin ids:\n{np.unique(df_gene_snp.bin_id)}")

    if df_gene_snp.bin_id.isnull().any():
        logger.warning(f"NaN bin id detected.")

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

    assert (
        df_gene_snp["bin_id"].nunique(dropna=True) == single_X.shape[0]
    ), f"{df_gene_snp['bin_id'].notna().sum()} != {single_X.shape[0]}"
    assert df_gene_snp["bin_id"].nunique(dropna=True) == sum(
        lengths
    ), f"{df_gene_snp['bin_id'].notna().sum()} != {sum(lengths)}"

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
