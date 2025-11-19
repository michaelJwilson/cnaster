import logging

import numpy as np
from cnaster.hmm import hmm_sitewise, pipeline_baum_welch
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.config import get_global_config

logger = logging.getLogger(__name__)


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
    
    # NB loop over initial clones.
    for i in range(n_clones):
        logger.info(f"Solving for phasing of initial clone {i} of {n_clones}.")

        # NB assumes BAF = 0.5 for insufficient snp umi count; initial binning chosen so this is not the case
        #    for pseudobulk of all spots?
        if np.sum(total_bb_RD[:, i]) < min_snpumi:
            logger.warning(f"Insufficient SNP UMI to infer BAF, assuming 0.5;")
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

            # NB MAP estimate of state given log posterior; pred. > n_states indicates switch-error.
            pred = np.argmax(res["log_gamma"], axis=0)

            # TODO calculate empirical switch error rate.
            
            # NB BAF by mirroring by inferred haplotype - assumed baf is not e.g. minor, but initialization
            #    dependent.
            this_baf_profiles = np.where(
                pred < n_states,
                res["new_p_binom"][pred % n_states, 0],
                1.0
                - res["new_p_binom"][
                    pred % n_states, 0
                ],  # BAF evidence for switch-error so flip.
            )

            # NB TODO attractor to 0.5 if sufficiently close, independent of coverage.
            EPS_BAF = 0.05  # MAGIC
            
            assumed_normal = np.abs(this_baf_profiles - 0.5) < EPS_BAF            
            this_baf_profiles[assumed_normal] = 0.5

            # NB solved for baf_profile of this clone, mitigating switch errors.
            baf_profiles[i, :] = this_baf_profiles
            
    # NB assumed minor baf profile.
    minor_baf_profiles = np.where(baf_profiles < 0.5, baf_profiles, 1.0 - baf_profiles)

    # NB compute population-level BAF with weighted mean by clone size.
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

    logger.info(f"Found non-normal population BAF to be:\n{np.unique(population_baf[population_baf != 0.5])}")

    # NB makes sense: phasing determined with all clones; copy state BAF phased appropriately.
    phase_indicator = population_baf < 0.5
    refined_lengths = []
    cumlen = 0

    config = get_global_config()
    BAF_CHANGE_THRESHOLD = config.phasing.baf_change_threshold # MAGIC
    MIN_SEGMENT_SIZE = config.phasing.min_new_segment_size

    # NB le is the number of blocks per contig.
    for le in lengths:
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
                refined_lengths.append(i - s)
                s = i

        refined_lengths.append(le - s)
        cumlen += le

    # NB expect to unpack 22 per-contig lengths of N segments per contig, to len(refined_lengths) = sum(lengths).
    refined_lengths = np.array(refined_lengths)

    logger.info(
        f"Solved for {len(refined_lengths)} phase-refined lengths given {len(lengths)} input lengths with sum={sum(lengths)}."
    )

    return phase_indicator, refined_lengths
