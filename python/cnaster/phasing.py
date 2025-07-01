from cnaster.pseduobulk import merge_pseudobulk_by_index

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
    EPS_BAF = 0.05
    
    if single_tumor_prop is None:
        tumor_prop = None
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(
            single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index
        )
    else:
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            initial_clone_index,
            single_tumor_prop,
            threshold=threshold,
        )

    # pseudobulk HMM for phase_prob
    baf_profiles = np.zeros((X.shape[2], X.shape[0]))
    pred_cnv = np.zeros((X.shape[2], X.shape[0]))
    for i in range(X.shape[2]):
        if np.sum(total_bb_RD[:, i]) < min_snpumi:
            baf_profiles[i, :] = 0.5
        else:
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
            #
            pred = np.argmax(res["log_gamma"], axis=0)
            this_baf_profiles = np.where(
                pred < n_states,
                res["new_p_binom"][pred % n_states, 0],
                1 - res["new_p_binom"][pred % n_states, 0],
            )
            this_baf_profiles[np.abs(this_baf_profiles - 0.5) < EPS_BAF] = 0.5
            baf_profiles[i, :] = this_baf_profiles
            pred_cnv[i, :] = pred % n_states

    if single_tumor_prop is None:
        n_total_spots = np.sum([len(x) for x in initial_clone_index])
        population_baf = (
            np.array([1.0 * len(x) / n_total_spots for x in initial_clone_index])
            @ baf_profiles
        )
    else:
        n_total_spots = np.sum(
            [len(x) * tumor_prop[i] for i, x in enumerate(initial_clone_index)]
        )
        population_baf = (
            np.array(
                [
                    1.0 * len(x) * tumor_prop[i] / n_total_spots
                    for i, x in enumerate(initial_clone_index)
                ]
            )
            @ baf_profiles
        )
    adj_baf_profiles = np.where(baf_profiles < 0.5, baf_profiles, 1 - baf_profiles)
    phase_indicator = population_baf < 0.5
    refined_lengths = []
    cumlen = 0
    for le in lengths:
        s = 0
        for i in range(le):
            if i > s + 10 and np.any(
                np.abs(
                    adj_baf_profiles[:, i + cumlen]
                    - adj_baf_profiles[:, i + cumlen - 1]
                )
                > 0.1
            ):
                refined_lengths.append(i - s)
                s = i
        refined_lengths.append(le - s)
        cumlen += le
    refined_lengths = np.array(refined_lengths)
    return phase_indicator, refined_lengths
