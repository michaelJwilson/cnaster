import numpy as np
from cnaster.config import get_global_config
from cnaster.config import start_time
from cnaster.logger import get_logger
from cnaster.reference import get_reference_recomb_rates

logger = get_logger(__name__, start_time=start_time)

# recomb_rates = get_reference_recomb_rates(config.references.geneticmap_file)
#
# write_fig(
#     f"{plots_dir}/recombination_rates.pdf",
#     plot_recombination_rates(recomb_rates),
#     transparent=True,bbox_inches="tight"
# )


def compute_numbat_phase_switch_prob(
    position_cM, chr_pos_vector, nu=1.0, min_prob=None
):
    """
    Attributes
    ----------
    position_cM : array, (number SNP positions)
        Centimorgans of SNPs located at each entry of position_cM.

    chr_pos_vector : list of pairs
        list of (chr, pos) pairs of SNPs. It is used to identify start of a new chr.
    """
    if min_prob is None:
        min_prob = get_global_config().phasing.min_prob

    logger.info_once(f"Computing numbat phase switch probabilities assuming nu={nu}.")
    logger.info(
        f"position_cM has {100. * np.mean(np.isnan(position_cM))}% NAN content."
    )

    phase_switch_prob = min_prob * np.ones(len(position_cM))

    for i, cm in enumerate(position_cM[:-1]):
        cm_next = position_cM[i + 1]

        # NB defaults to min_prob. in these cases.
        if (
            np.isnan(cm)
            or np.isnan(cm_next)
            or chr_pos_vector[i][0] != chr_pos_vector[i + 1][0]
        ):
            continue

        assert cm <= cm_next

        d = cm_next - cm

        # NB numbat definition;
        phase_switch_prob[i] = (1.0 - np.exp(-2.0 * nu * d)) / 2.0

    logger.info_once(f"Solved for max phase switch prob. = {np.max(phase_switch_prob)}")

    under_flowed = phase_switch_prob < min_prob

    logger.info(
        f"Reassigning under flowed phase_switch_prob. for {100. * np.mean(under_flowed):.6e}% given {min_prob} threshold."
    )

    phase_switch_prob[phase_switch_prob < min_prob] = min_prob

    return phase_switch_prob


def assign_centiMorgans(chr_pos_vector, ref_positions_cM):
    logger.info_once("Assigning centiMorgan recombination rates.")

    ref_chrom = np.array(ref_positions_cM.chrom).astype(int)
    ref_pos = np.array(ref_positions_cM.pos)
    ref_cm = np.array(ref_positions_cM.pos_cm)

    # TODO
    chr_pos_vector.sort()

    # NB find the centimorgan values (linear interpolation between (k-1)-th and k-th rows of table.
    position_cM = np.ones(len(chr_pos_vector)) * np.nan
    k = 0

    for i, x in enumerate(chr_pos_vector):
        chrname = x[0]

        # NB? start position.
        pos = x[1]

        # NB fast-forward k to this contig and start.
        while k < len(ref_chrom) and (
            ref_chrom[k] < chrname or (ref_chrom[k] == chrname and ref_pos[k] < pos)
        ):
            k += 1

        if k < len(ref_chrom) and (ref_chrom[k] == chrname) and (ref_pos[k] >= pos):
            if k > 0 and ref_chrom[k - 1] == chrname:
                position_cM[i] = ref_cm[k - 1] + (pos - ref_pos[k - 1]) / (
                    ref_pos[k] - ref_pos[k - 1]
                ) * (ref_cm[k] - ref_cm[k - 1])
            else:
                # NB Extrapolation from Chromosome Start
                position_cM[i] = (pos - 0) / (ref_pos[k] - 0) * (ref_cm[k] - 0)
        else:
            # NB Extrapolation Beyond Last Reference Point
            position_cM[i] = ref_cm[k - 1]

    # NB given input [(chr1, start1), (chr1, end1), (chr2, start2), (chr2, end2), ...])
    #    return positions in centiMorgan.
    return position_cM


def get_sitewise_transmat(df_gene_snp, geneticmap_file, nu, logphase_shift):
    """
    Phase switch probability from recombination rate / genetic distance [cM].
    """
    logger.info(
        f"Constructing sitewise transition matrix for phasing given recombination rates."
    )

    # NB define recombination rates.
    ref_positions_cM = get_reference_recomb_rates(geneticmap_file)

    # NB sorted contig,start per block.
    sorted_chr_pos_first = df_gene_snp.groupby("block_id").agg(
        {"CHR": "first", "START": "first"}
    )

    sorted_chr_pos_last = df_gene_snp.groupby("block_id").agg(
        {"CHR": "last", "END": "last"}
    )

    if sorted_chr_pos_first.index.isna().any():
        logger.warning(f"Found ill-defined group with None entries for group.")

    # NB dataframe to list.
    sorted_chr_pos_first = list(
        zip(sorted_chr_pos_first.CHR.to_numpy(), sorted_chr_pos_first.START.to_numpy())
    )

    sorted_chr_pos_last = list(
        zip(sorted_chr_pos_last.CHR.to_numpy(), sorted_chr_pos_last.END.to_numpy())
    )

    # NB [(chr1, start1), (chr1, end1), (chr2, start2), (chr2, end2), ...]) construct ...
    tmp_sorted_chr_pos = [
        val for pair in zip(sorted_chr_pos_first, sorted_chr_pos_last) for val in pair
    ]

    # NB positions in cM of [(chr1, start1), (chr1, end1), (chr2, start2), (chr2, end2), ...])
    position_cM = assign_centiMorgans(tmp_sorted_chr_pos, ref_positions_cM)

    # NB tmp_sorted_chr_pos used to identify chromosome switches.
    phase_switch_prob = compute_numbat_phase_switch_prob(
        position_cM, tmp_sorted_chr_pos, nu
    )

    # NB transition matrix for phasing.
    log_sitewise_transmat = np.minimum(
        np.log(0.5), np.log(phase_switch_prob) - logphase_shift
    )

    # NB positions -> pairs by sampling at rate 2.
    log_sitewise_transmat = log_sitewise_transmat[
        np.arange(1, len(log_sitewise_transmat), 2)
    ]

    logger.info(
        f"Solved for (recombination based) sitewise transition matrix for phasing with shape={log_sitewise_transmat.shape}."
    )

    # NB returns array.
    return log_sitewise_transmat


"""
def compute_numbat_phase_switch_prob(
    position_cM, chr_pos_vector, nu=1.0, min_prob=None
):
    if min_prob is None:
        min_prob = get_global_config().phasing.min_prob

    logger.info_once(f"Computing numbat phase switch probabilities assuming nu={nu}.")
    logger.info(
        f"position_cM has {100. * np.mean(np.isnan(position_cM))}% NAN content."
    )
    
    chr_pos_vector = np.asarray(chr_pos_vector)

    phase_switch_prob = np.full(len(position_cM), min_prob)

    cm = position_cM[:-1]
    cm_next = position_cM[1:]
    
    chr_curr = chr_pos_vector[:-1, 0]
    chr_next = chr_pos_vector[1:, 0]

    valid_mask = ~np.isnan(cm) & ~np.isnan(cm_next) & (chr_curr == chr_next)
    
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) > 0:
        d = cm_next[valid_mask] - cm[valid_mask]
        
        probs = (1.0 - np.exp(-2.0 * nu * d)) / 2.0
        phase_switch_prob[valid_indices] = probs

    logger.info_once(f"Solved for max phase switch prob. = {np.max(phase_switch_prob)}")

    under_flowed = phase_switch_prob < min_prob
    logger.info(
        f"Reassigning under flowed phase_switch_prob. for {100. * np.mean(under_flowed):.6e}% given {min_prob} threshold."
    )

    # Clamp the lower bound 
    phase_switch_prob = np.maximum(phase_switch_prob, min_prob)

    return phase_switch_prob


def assign_centiMorgans(chr_pos_vector, ref_positions_cM):
    logger.info_once("Assigning centiMorgan recombination rates.")

    chr_pos_vector = np.asarray(chr_pos_vector)

    ref_chrom = np.array(ref_positions_cM.chrom).astype(int)
    ref_pos = np.array(ref_positions_cM.pos)
    ref_cm = np.array(ref_positions_cM.pos_cm)

    position_cM = np.full(len(chr_pos_vector), np.nan)
    
    target_chrom = chr_pos_vector[:, 0]
    target_pos = chr_pos_vector[:, 1]

    for chrom in np.unique(target_chrom):
        t_mask = target_chrom == chrom
        t_pos = target_pos[t_mask]
        
        r_mask = ref_chrom == chrom
        if not np.any(r_mask):
            continue
            
        r_pos_chrom = ref_pos[r_mask]
        r_cm_chrom = ref_cm[r_mask]
        
        if r_pos_chrom[0] > 0:
            r_pos_chrom = np.insert(r_pos_chrom, 0, 0.0)
            r_cm_chrom = np.insert(r_cm_chrom, 0, 0.0)
            
        position_cM[t_mask] = np.interp(t_pos, r_pos_chrom, r_cm_chrom)

    return position_cM


def get_sitewise_transmat(df_gene_snp, geneticmap_file, nu, logphase_shift):
    logger.info(
        "Constructing sitewise transition matrix for phasing given recombination rates."
    )

    ref_positions_cM = get_reference_recomb_rates(geneticmap_file)

    grouped = df_gene_snp.groupby("block_id").agg(
        first_chr=("CHR", "first"),
        first_start=("START", "first"),
        last_chr=("CHR", "last"),
        last_end=("END", "last")
    )

    if grouped.index.isna().any():
        logger.warning("Found ill-defined group with None entries for group.")

    N = len(grouped)
    
    tmp_sorted_chr_pos = np.empty((2 * N, 2))
    tmp_sorted_chr_pos[0::2, 0] = grouped["first_chr"].to_numpy()
    tmp_sorted_chr_pos[0::2, 1] = grouped["first_start"].to_numpy()
    tmp_sorted_chr_pos[1::2, 0] = grouped["last_chr"].to_numpy()
    tmp_sorted_chr_pos[1::2, 1] = grouped["last_end"].to_numpy()

    position_cM = assign_centiMorgans(tmp_sorted_chr_pos, ref_positions_cM)

    phase_switch_prob = compute_numbat_phase_switch_prob(
        position_cM, tmp_sorted_chr_pos, nu
    )

    log_sitewise_transmat = np.minimum(
        np.log(0.5), np.log(phase_switch_prob) - logphase_shift
    )

    log_sitewise_transmat = log_sitewise_transmat[1::2]

    return log_sitewise_transmat
"""
