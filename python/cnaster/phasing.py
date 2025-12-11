import logging

import numpy as np
from collections import namedtuple
from cnaster.utils import cacher
from cnaster.hmm import hmm_sitewise, pipeline_baum_welch
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.config import get_global_config

logger = logging.getLogger(__name__)

@cacher("initial_phase.hdf5")
def initial_phase_given_partition(
    single_X,
    lengths,
    single_base_nb_mean,
    single_total_bb_RD,
    single_tumor_prop,
    initial_clone_index,
    n_states,
    log_sitewise_transmat,
    params,
    t,
    random_state,
    fix_NB_dispersion,
    shared_NB_dispersion,
    fix_BB_dispersion,
    shared_BB_dispersion,
    max_iter,
    tol,
    threshold,
    min_snpumi=2e3,
):
    # NB see https://github.com/raphael-group/CalicoST/blob/c1abcae3e3657e01e547ee4529e3b9d039221453/src/calicost/phasing.py#L50
    #    utilizes tumor_prop for baf_profile weighting only; not in hmm fitting.
    assert np.all(single_base_nb_mean == 0)

    # NB TODO attractor to 0.5 if sufficiently close, independent of coverage.
    EPS_BAF = 0.05  # MAGIC

    # NB on input phase_indicator is 0s by construction, up to phase switch errors TBD.
    logger.info(f"Starting phasing assuming {len(initial_clone_index)} clones.")

    # NB aggregate given initial clones.
    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        initial_clone_index,
        single_tumor_prop,
        threshold=threshold,
    )

    # NB (initial clones, segments).
    n_clones = X.shape[2]
    baf_profiles = np.zeros((n_clones, X.shape[0]))
    phase_profiles = np.zeros((n_clones, X.shape[0]), dtype=bool)

    cumulative_lengths = np.cumsum(lengths)

    for i in range(n_clones):
        logger.info(f"Solving for phasing of initial clone {i} of {n_clones}.")

        # NB assumes BAF = 0.5 for insufficient snp umi count; initial binning chosen so this is not the case
        #    for pseudobulk of all spots?
        if np.sum(total_bb_RD[:, i]) < min_snpumi:
            logger.warning(f"Insufficient snp umi to infer BAF, assuming 0.5;")
            baf_profiles[i, :] = 0.5
        else:
            # NB phasing of a single clone; independent BAF values.
            res = pipeline_baum_welch(
                None,
                X[:, :, i : (i + 1)],
                lengths,
                n_states,
                base_nb_mean[:, i : (i + 1)],
                total_bb_RD[:, i : (i + 1)],
                log_sitewise_transmat,
                tumor_prop=tumor_prop, # NB calicost assumes tumor_prop is None
                hmmclass=hmm_sitewise,
                params=params,
                t=t,
                random_state=random_state,
                only_minor=True,
                fix_NB_dispersion=fix_NB_dispersion,
                shared_NB_dispersion=shared_NB_dispersion,
                fix_BB_dispersion=fix_BB_dispersion,
                shared_BB_dispersion=shared_BB_dispersion,
                is_diag=True,
                init_log_mu=None,
                init_p_binom=None,
                init_alphas=None,
                init_taus=None,
                max_iter=max_iter,
                tol=tol,
            )

            if not np.all(res["new_p_binom"] <= 0.5 + EPS_BAF):
                logger.warning(f"Found best-fit phased baf > 0.5 += eps.")

            # NB MAP estimate of state given log posterior; pred. > n_states indicates switch-error.
            pred = np.argmax(res["log_gamma"], axis=0)

            # NB by initialization, baf p_binom is minor (<0.5); states > n_states are phase flips
            #    corresponding to 1 - p_binom.
            #
            #    This has the derived consequence that all data > 0.5 is "flipped", irrespective of
            #    whether there was a switch error, i.e. a run of p, followed by 1-p, or vice versa. 
            this_baf_profiles = np.where(
                pred < n_states,
                res["new_p_binom"][pred % n_states, 0],
                1.0
                - res["new_p_binom"][
                    pred % n_states, 0
                ],
            )

            assumed_normal = np.abs(this_baf_profiles - 0.5) < EPS_BAF
            this_baf_profiles[assumed_normal] = 0.5

            # NB model of the baf profile (as copy states) for all clones, phased.
            baf_profiles[i, :] = this_baf_profiles

            phase_profiles[i, :] = pred < n_states
            phase_profiles[i, res["new_p_binom"][pred % n_states, 0] > 0.5] = True
            phase_profiles[i, assumed_normal] = True
            """
            valid = ~assumed_normal
            baf_switches = np.where(
                valid[:-1] 
                & (np.abs(np.diff(this_minor_baf_profiles)) > EPS_BAF)
            )[0]

            phase_switches = np.where( 
                valid[:-1] 
                & (np.abs(this_baf_profiles[:-1] - (1. - this_baf_profiles[1:])) < EPS_BAF)
            )[0]

            n_total_bins = len(pred)

            # NB log best-fit mu, p for this clone
            logger.info(f"Clone {i} mu=\n{np.exp(res['new_log_mu'])}\nand\np={res['new_p_binom']}")

            for phase_switch in phase_switches:
                remaining_baf_switches = baf_switches[baf_switches > phase_switch]
                next_baf_change = remaining_baf_switches[0] if len(remaining_baf_switches) > 0 else n_total_bins

                remaining_lengths = cumulative_lengths[cumulative_lengths > phase_switch]
                next_end = remaining_lengths[0] if len(remaining_lengths) > 0 else n_total_bins

                num_segments_to_flip = min(1 + next_baf_change, next_end) - phase_switch

                contig = 1 + np.searchsorted(cumulative_lengths, phase_switch, side='right')
                baf_val = this_baf_profiles[phase_switch]
                next_baf = this_baf_profiles[1 + next_baf_change] if 1 + next_baf_change < n_total_bins else np.nan

                logger.info(
                    f"Clone {i}, chr{contig} phase switch for baf={baf_val:.6f} and {num_segments_to_flip} segments (next_baf={next_baf:.6f})."
                )
            """

    # NB corresponds to the baf profile as realized by res["new_p_binom"][pred % n_states, 0].
    minor_baf_profiles = np.where(baf_profiles < 0.5, baf_profiles, 1.0 - baf_profiles)

    # NB compute population-level BAF, weighted by clone fraction.
    if single_tumor_prop is None:
        num_spots_per_clone = [len(x) for x in initial_clone_index]
        n_total_spots = np.sum(num_spots_per_clone)

        population_baf = (
            np.array([1.0 * len(x) / n_total_spots for x in initial_clone_index])
            @ baf_profiles
        )
    else:
        # NB tumor_prop is the mean of each clone.
        n_total_spots = np.sum(
            [len(x) * tumor_prop[i] for i, x in enumerate(initial_clone_index)]
        )

        # NB? tumor_prop for pseudo-bulk.
        population_baf = (
            np.array(
                [
                    1.0 * len(x) * tumor_prop[i] / n_total_spots
                    for i, x in enumerate(initial_clone_index)
                ]
            )
            @ baf_profiles
        )

    # NB behaviour is to force a flip of the data is the population model baf >0.5;
    #    this is shared by all spots/clones, so mirrored events are conserved, e.g.
    #   (2,1) & (1,2) -> (1,2) & (2,1),
    #
    # TODO HACK < -> <= to reduce flips for EPS_BAF.
    config = get_global_config()
    BAF_CHANGE_THRESHOLD = config.phasing.baf_change_threshold
    MIN_SEGMENT_SIZE = config.phasing.min_new_segment_size

    if config.run.legacy:
        phase_indicator = population_baf < 0.5
        logger.info(f"Legacy phase indicator assumed {np.count_nonzero(phase_indicator)}/{len(phase_indicator)} (mean={np.mean(phase_indicator)}) switches.")

    else:
        phase_indicator = population_baf <= 0.5
        
        # TODO HACK flipped where false.                                                                                                                                                                                                                 
        # phase_indicator = np.all(phase_profiles, axis=0)

        logger.info(f"Phase indicator assumes {np.count_nonzero(phase_indicator)}/{len(phase_indicator)} (mean={np.mean(phase_indicator)}) switches.")
                
    refined_lengths = []
    cumlen = 0

    # NB TODO?  this can only be necessary if phase indicator does not correctly capture all switches,
    #           and potentially allows merges that should be excluded based on the BAF.  
    # 
    # le is the number of blocks per contig.
    for ii, le in enumerate(lengths):
        s = 0

        for i in range(le):
            # NB min. segment size of 10
            if i > s + MIN_SEGMENT_SIZE and np.any(
                np.abs(
                    minor_baf_profiles[:, i + cumlen]
                    - minor_baf_profiles[:, i + cumlen - 1]
                )
                >= BAF_CHANGE_THRESHOLD
            ):
                # NB new blocks are a min. size and set by change in BAF.
                logger.warning(f"Forced a block boundary at contig {1 + ii} pos {i} given dBAF={np.abs(minor_baf_profiles[:, i + cumlen] - minor_baf_profiles[:, i + cumlen - 1]).max()}.")
                refined_lengths.append(i - s)
                s = i

        # NB force a stop at contig end.
        refined_lengths.append(le - s)
        cumlen += le

    # NB expect to unpack 22 per-contig lengths of N segments per contig, to len(refined_lengths) = sum(lengths).
    refined_lengths = np.array(refined_lengths)

    logger.info(
        f"Solved for {len(refined_lengths)} phase-refined lengths given {len(lengths)} input lengths with sum={sum(lengths)}."
    )

    # NB return named tuple PhaseSummary
    PhaseSummary = namedtuple("PhaseSummary", ["phase_indicator", "refined_lengths"])

    return PhaseSummary(phase_indicator=phase_indicator, refined_lengths=refined_lengths)
