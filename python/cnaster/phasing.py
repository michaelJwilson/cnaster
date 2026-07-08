import time
import numpy as np

# from collections import namedtuple
# from cnaster.utils import cacher
# from cnaster.hmm import pipeline_baum_welch, hmm_sitewise
from cnaster.hmm_phased import hmm_phased
# from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.hmrf_utils import clone_stack_obs
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.config import get_global_config
from cnaster.hmm_initialize import gmm_init, cna_mixture_init
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

# cell_snp_Aallele, cell_snp_Ballele = perturb_phase(
#     cell_snp_Aallele, cell_snp_Ballele, 0.1
# )


# NB mirrors calicost.phasing.initial_phase_given_partition;
# @cacher("initial_phase.hdf5")
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
    known_normal=False,
    hmm_initializer=cna_mixture_init,
    # min_snpumi=2e3,
):
    """
    Phasing routine:
        -  aggregates by the provided clone assignment, initial_clone_index.
        -  initilizes minor only p_binom with gmm.
        -  runs baum welch to estimate the HMM parameters (i.e. dispersions) using __minor__ baf only.
        -  assumes low tolerance on HMM (run_cnaster defined).
        -  runs __phased__ baum welch per clone to estimate parameters and phasing, with no state sharing.
        -  builds model baf profiles for all clones, together with a phase vector.
        -  decides the phase vector by majority vote across clones, ignoring normal-like segments (baf ~ 0.5).
        -  phase vector will be used to aggregate segments, based on required counts.
        -  refines the input lengths by splitting segments where the minor BAF changes by more than BAF_CHANGE_THRESHOLD,
           and the segment is larger than MIN_SEGMENT_SIZE.
    """
    assert np.all(single_base_nb_mean == 0)

    # NB TODO attractor to 0.5 if sufficiently close, independent of coverage.
    EPS_BAF = 0.1  # MAGIC

    start_time = time.time()

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
    '''
    # NB force baf < 0.5 by taking (1. - single_X[:,1,:]) where single_X[:,1,:]) / single_total_bb_RD > 0.5
    baf = X[:, 1, :] / total_bb_RD

    minor_X = np.zeros_like(X)
    minor_X[:, 0, :] = X[:, 0, :]
    minor_X[:, 1, :] = np.where(baf > 0.5, total_bb_RD - X[:, 1, :], X[:, 1, :])
    '''
    # NB (initial clones, segments).
    n_obs, _, n_clones = X.shape

    (
        clone_stack_X,
        clone_stack_base_nb_mean,
        clone_stack_total_bb_RD,
        clone_stack_lengths,
        clone_stack_sitewise_transmat,
        _,
    ) = clone_stack_obs(
        X, base_nb_mean, total_bb_RD, lengths, log_sitewise_transmat, tumor_prop
    )

    # TODO
    init_log_mu, init_p_binom = hmm_initializer(
        n_states,
        clone_stack_X,
        clone_stack_base_nb_mean,
        clone_stack_total_bb_RD,
        params,
        random_state=random_state,
        in_log_space=False,
        only_minor=True,
    )

    # >>>>>>>>
    # TODO rename model_baf_profiles
    res = hmm_phased(params="sp", t=t).run_baum_welch_nb_bb(
        clone_stack_X,
        clone_stack_lengths,
        n_states,
        clone_stack_base_nb_mean,
        total_bb_RD=clone_stack_total_bb_RD,
        log_sitewise_transmat=clone_stack_sitewise_transmat,
        fix_NB_dispersion=fix_NB_dispersion,
        shared_NB_dispersion=shared_NB_dispersion,
        fix_BB_dispersion=fix_BB_dispersion,
        shared_BB_dispersion=shared_BB_dispersion,
        init_log_mu=init_log_mu,
        init_p_binom=init_p_binom,
        max_iter=max_iter,
        tol=tol,
    )

    # n_clones = len(np.unique(res["new_assignment"]))

    # NB includes (combinatorial-space) phase.
    pred = np.argmax(res["log_gamma"], axis=0)
    
    # NB vectorize split into clone-wise array shape: (n_clones, n_obs)
    pred = pred.reshape(n_clones, n_obs)

    # Vectorized model_baf_profiles construction
    base_states = pred % n_states
    phase_mask = pred < n_states
    base_bafs = res["new_p_binom"][base_states, 0]
    
    model_baf_profiles = np.where(phase_mask, base_bafs, 1.0 - base_bafs)
    minor_baf_profiles = np.where(model_baf_profiles < 0.5, model_baf_profiles, 1.0 - model_baf_profiles)

    assumed_normal = np.abs(model_baf_profiles - 0.5) < EPS_BAF

    # Explicitly cast to integer so assigning -1 actually works
    phase_profiles = phase_mask.astype(np.int8)

    # NB do not define phase for normal-like segments (for this clone).
    phase_profiles[assumed_normal] = -1
    
    # Define phase votes
    phase_votes = phase_profiles[1:, :] if known_normal else phase_profiles

    # Vectorize the voting logic (avoids python loop over n_obs)
    valid_mask = phase_votes != -1
    valid_counts = np.sum(valid_mask, axis=0)
    valid_sums = np.sum(np.where(valid_mask, phase_votes, 0), axis=0)

    phase_indicator = np.zeros(n_obs, dtype=int)
    has_votes = valid_counts > 0
    phase_indicator[has_votes] = (valid_sums[has_votes] / valid_counts[has_votes]) >= 0.5
    # <<<<<<<<<<<<

    '''
    # NB initial dispersion estimate assuming no phasing.
    res = pipeline_baum_welch(
        None,
        clone_stack_minor_X,
        clone_stack_lengths,
        n_states,
        clone_stack_base_nb_mean,
        clone_stack_total_bb_RD,
        clone_stack_sitewise_transmat,
        clone_stack_tumor_prop,
        hmmclass=hmm_nophasing,
        params=params,
        t=t,
        random_state=random_state,
        fix_NB_dispersion=fix_NB_dispersion,
        shared_NB_dispersion=shared_NB_dispersion,
        fix_BB_dispersion=fix_BB_dispersion,
        shared_BB_dispersion=shared_BB_dispersion,
        init_log_mu=init_log_mu,
        init_p_binom=init_p_binom,
        max_iter=max_iter,
        tol=tol,
    )

    # TODO rename model_baf_profiles
    baf_profiles, phase_profiles = np.zeros((n_clones, X.shape[0])), np.zeros(
        (n_clones, X.shape[0])
    )

    for i in range(n_clones):
        logger.info(f"Solving for phasing of initial clone {i} of {n_clones}.")

        # NB assumes BAF = 0.5 for insufficient snp umi count; initial binning chosen so this is not the case
        #    for pseudobulk of all spots?
        #
        # NB phasing of a single clone; independent BAF values.
        """
        res = pipeline_baum_welch(
            None,
            X[:, :, i : (i + 1)],
            lengths,
            n_states,
            base_nb_mean[:, i : (i + 1)],
            total_bb_RD[:, i : (i + 1)],
            log_sitewise_transmat,
            tumor_prop=tumor_prop, # NB calicost assumes tumor_prop is None
            hmmclass=hmm_sitewise, # NB assumes hmm_sitewise
            params="", # NB does not solve for p(!) or s.
            t=t,
            random_state=random_state,
            only_minor=True,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            is_diag=True,
            init_log_mu=init_log_mu,
            init_p_binom=init_p_binom,
            init_alphas=res["new_alphas"],
            init_taus=res["new_taus"],
            max_iter=max_iter,
            tol=tol,
        )
        """
        res = hmm_phased(params="sp", t=t).run_baum_welch_nb_bb(
            X[:, :, i : (i + 1)],
            lengths,
            n_states,
            base_nb_mean[:, i : (i + 1)],
            total_bb_RD[:, i : (i + 1)],
            log_sitewise_transmat,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            init_log_mu=init_log_mu,
            init_p_binom=init_p_binom,
            init_alphas=res["new_alphas"],
            init_taus=res["new_taus"],
            max_iter=max_iter,
            tol=tol,
        )

        # NB MAP estimate of state given log posterior; pred. > n_states indicates switch-error.
        pred = np.argmax(res["log_gamma"], axis=0)

        baf_profiles[i, :] = np.where(
            pred < n_states,
            res["new_p_binom"][pred % n_states, 0],
            1.0 - res["new_p_binom"][pred % n_states, 0],
        )

        assumed_normal = np.abs(baf_profiles[i, :] - 0.5) < EPS_BAF

        phase_profiles[i, :] = pred < n_states

        # NB do not define phase for normal-like segments (for this clone).
        phase_profiles[i, assumed_normal] = -1
    
    minor_baf_profiles = np.where(baf_profiles < 0.5, baf_profiles, 1.0 - baf_profiles)
    
    # NB phase_indicator is the majority vote across clones; assuming normal is clone 0.
    phase_indicator = np.zeros(X.shape[0], dtype=int)
    phase_votes = phase_profiles[1:, :] if known_normal else phase_profiles[:, :]

    for idx in range(X.shape[0]):
        valid_votes = phase_votes[:, idx][phase_votes[:, idx] != -1]
        if valid_votes.size == 0:
            phase_indicator[idx] = 0
        else:
            phase_indicator[idx] = np.mean(valid_votes) >= 0.5
    '''
    # TODO HACK < -> <= to reduce flips for EPS_BAF.
    config = get_global_config()
    BAF_CHANGE_THRESHOLD = config.phasing.baf_change_threshold
    MIN_SEGMENT_SIZE = config.phasing.min_new_segment_size

    refined_lengths = []
    cumlen = 0

    # NB TODO?  this can only be necessary if phase indicator does not correctly capture all switches,
    #           and potentially allows merges that should be excluded based on the BAF.
    #
    # le is the number of blocks per contig.
    for ii, le in enumerate(lengths):
        s = 0

        # NB we loop through the blocks on this contig.
        for i in range(le):
            # NB if there's a (BAF_CHANGE_THRESHOLD) step in the minor baf (and min. segment size of 10) the contig partitioning of blocks
            #    is refined to split the block at the minor baf step.
            if i > s + MIN_SEGMENT_SIZE and np.any(
                np.abs(
                    minor_baf_profiles[:, i + cumlen]
                    - minor_baf_profiles[:, i + cumlen - 1]
                )
                >= BAF_CHANGE_THRESHOLD
            ):
                # NB new blocks are a min. size and set by change in BAF - conserved phase (minor baf state) would imply segmentation, but evidence baf changes.
                logger.warning(
                    f"Forced a block boundary at contig {1 + ii} pos {i} despite conserved phase (minor baf state), given a baf switch of {np.abs(minor_baf_profiles[:, i + cumlen] - minor_baf_profiles[:, i + cumlen - 1]).max():.4f}."
                )
                refined_lengths.append(i - s)
                s = i

        # NB force a stop at contig end.  Guranteed to have more blocks than the original (number of contigs).
        refined_lengths.append(le - s)
        cumlen += le

    # NB expect to unpack 22 per-contig lengths of N segments per contig, to len(refined_lengths) = sum(lengths).
    refined_lengths = np.array(refined_lengths)

    end_time = time.time()

    logger.info(
        f"Solved for {len(refined_lengths)} phase-refined lengths () given {len(lengths)} input lengths with sum={sum(lengths)} in {(end_time - start_time):.2f} seconds."
    )

    return res, phase_indicator, refined_lengths

    # NB return named tuple PhaseSummary
    # PhaseSummary = namedtuple("PhaseSummary", ["phase_indicator", "refined_lengths"])

    # return PhaseSummary(phase_indicator=phase_indicator, refined_lengths=refined_lengths)
