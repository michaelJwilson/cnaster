import logging
import scipy

from cnaster.hmm_emission import Weighted_BetaBinom
from cnaster.reference import get_reference_recomb_rates
from cnaster.recomb import assign_centiMorgans, compute_numbat_phase_switch_prob


logger = logging.getLogger(__name__)


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

    phase_switch_prob = compute_phase_switch_probability_position(
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
