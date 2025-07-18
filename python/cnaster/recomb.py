import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_numbat_phase_switch_prob(
    position_cM, chr_pos_vector, nu=1.0, min_prob=1.0e-20
):
    """
    Attributes
    ----------
    position_cM : array, (number SNP positions)
        Centimorgans of SNPs located at each entry of position_cM.

    chr_pos_vector : list of pairs
        list of (chr, pos) pairs of SNPs. It is used to identify start of a new chr.
    """
    logger.info("Computing numbat phase switch probabilities")

    phase_switch_prob = min_prob * np.ones(len(position_cM))

    for i, cm in enumerate(position_cM[:-1]):
        cm_next = position_cM[i + 1]

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

    phase_switch_prob[phase_switch_prob < min_prob] = min_prob

    return phase_switch_prob


def assign_centiMorgans(chr_pos_vector, ref_positions_cM):
    logger.info("Assigning centiMorgan recombination rates.")

    ref_chrom = np.array(ref_positions_cM.chrom).astype(int)
    ref_pos = np.array(ref_positions_cM.pos)
    ref_cm = np.array(ref_positions_cM.pos_cm)

    # TODO
    chr_pos_vector.sort()

    # find the centimorgan values (interpolate between (k-1)-th and k-th rows
    # in centimorgan tables.
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

    return position_cM
