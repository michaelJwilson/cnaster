import numpy as np


def hmrfmix_reassignment_posterior_concatenate(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    single_tumor_prop,
    res,
    smooth_mat,
    adjacency_mat,
    prev_assignment,
    sample_ids,
    log_persample_weights,
    spatial_weight,
    hmmclass=hmm_sitewise,
    return_posterior=False,
):
    N = single_X.shape[2]
    
    n_obs = single_X.shape[0]
    n_clones = np.max(prev_assignment) + 1
    n_states = res["new_p_binom"].shape[0]

    # NB this will be the node potential for all (spot, clone).
    single_llf = np.zeros((N, n_clones))
    
    new_assignment = copy.copy(prev_assignment)

    # NB baseline expression.
    lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)

    # NB shift baseline mu according to Tn?
    if np.sum(single_base_nb_mean) > 0:
        logmu_shift = []

        for c in range(n_clones):
            this_pred_cnv = (
                np.argmax(
                    res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)], axis=0
                )
                % n_states
            )

            logmu_shift.append(
                scipy.special.logsumexp(
                    res["new_log_mu"][this_pred_cnv, :] + np.log(lambd).reshape(-1, 1),
                    axis=0,
                )
            )

        logmu_shift = np.vstack(logmu_shift)
        
        kwargs = {
            "logmu_shift": logmu_shift,
            "sample_length": np.ones(n_clones, dtype=int) * n_obs,
        }
    else:
        kwargs = {}

    # NB spot posterior under all clones.
    posterior = np.zeros((N, n_clones))

    for i in range(N):
        # NB aggregate across connected components implied by smooth mat.
        idx = smooth_mat[i, :].nonzero()[1]
        idx = idx[~np.isnan(single_tumor_prop[idx])]

        for c in range(n_clones):
            (
                tmp_log_emission_rdr,
                tmp_log_emission_baf,
            ) = hmmclass.compute_emission_probability_nb_betabinom_mix(
                np.sum(single_X[:, :, idx], axis=2, keepdims=True),
                np.sum(single_base_nb_mean[:, idx], axis=1, keepdims=True),
                res["new_log_mu"],
                res["new_alphas"],
                np.sum(single_total_bb_RD[:, idx], axis=1, keepdims=True),
                res["new_p_binom"],
                res["new_taus"],
                np.ones((n_obs, 1)) * np.mean(single_tumor_prop[idx]),
                **kwargs,
            )

            if (
                np.sum(single_base_nb_mean[:, i : (i + 1)] > 0) > 0
                and np.sum(single_total_bb_RD[:, i : (i + 1)] > 0) > 0
            ):
                ratio_nonzeros = (
                    1.0
                    * np.sum(single_total_bb_RD[:, i : (i + 1)] > 0)
                    / np.sum(single_base_nb_mean[:, i : (i + 1)] > 0)
                )

                single_llf[i, c] = ratio_nonzeros * np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_rdr[:, :, 0]
                        + res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)],
                        axis=0,
                    )
                ) + np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_baf[:, :, 0]
                        + res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)],
                        axis=0,
                    )
                )
            else:
                single_llf[i, c] = np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_rdr[:, :, 0]
                        + res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)],
                        axis=0,
                    )
                ) + np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_baf[:, :, 0]
                        + res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)],
                        axis=0,
                    )
                )
        w_node = single_llf[i, :]
        w_node += log_persample_weights[:, sample_ids[i]]
        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i, :].nonzero()[1]:
            # w_edge[new_assignment[j]] += 1
            w_edge[new_assignment[j]] += adjacency_mat[i, j]
        new_assignment[i] = np.argmax(w_node + spatial_weight * w_edge)
        #
        posterior[i, :] = np.exp(
            w_node
            + spatial_weight * w_edge
            - scipy.special.logsumexp(w_node + spatial_weight * w_edge)
        )

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum(
            spatial_weight
            * np.sum(
                new_assignment[adjacency_mat[i, :].nonzero()[1]] == new_assignment[i]
            )
        )
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf


def hmrfmix_concatenate_pipeline(
    outdir,
    prefix,
    single_X,
    lengths,
    single_base_nb_mean,
    single_total_bb_RD,
    single_tumor_prop,
    initial_clone_index,
    n_states,
    log_sitewise_transmat,
    coords=None,
    smooth_mat=None,
    adjacency_mat=None,
    sample_ids=None,
    max_iter_outer=5,
    nodepotential="max",
    hmmclass=hmm_sitewise,
    params="stmp",
    t=1 - 1e-6,
    random_state=0,
    init_log_mu=None,
    init_p_binom=None,
    init_alphas=None,
    init_taus=None,
    fix_NB_dispersion=False,
    shared_NB_dispersion=True,
    fix_BB_dispersion=False,
    shared_BB_dispersion=True,
    is_diag=True,
    max_iter=100,
    tol=1e-4,
    unit_xsquared=9,
    unit_ysquared=3,
    spatial_weight=1.0 / 6,
    tumorprop_threshold=0.5,
):
    n_obs, _, n_spots = single_X.shape
    n_clones = len(initial_clone_index)

    # spot adjacency matric
    assert not (coords is None and adjacency_mat is None)

    if adjacency_mat is None:
        adjacency_mat = compute_adjacency_mat(coords, unit_xsquared, unit_ysquared)

    if sample_ids is None:
        sample_ids = np.zeros(n_spots, dtype=int)
        n_samples = len(np.unique(sample_ids))
    else:
        unique_sample_ids = np.unique(sample_ids)
        n_samples = len(unique_sample_ids)
        tmp_map_index = {unique_sample_ids[i]: i for i in range(len(unique_sample_ids))}
        sample_ids = np.array([tmp_map_index[x] for x in sample_ids])

    log_persample_weights = np.ones((n_clones, n_samples)) * (-np.log(n_clones))

    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        initial_clone_index,
        single_tumor_prop,
        threshold=tumorprop_threshold,
    )

    # baseline proportion of UMI counts
    lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)

    # initialize HMM parameters by GMM
    if (init_log_mu is None) or (init_p_binom is None):
        init_log_mu, init_p_binom = initialization_by_gmm(
            n_states,
            np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
                -1, 2, 1
            ),
            base_nb_mean.flatten("F").reshape(-1, 1),
            total_bb_RD.flatten("F").reshape(-1, 1),
            params,
            random_state=random_state,
            in_log_space=False,
            only_minor=False,
        )

    # initialization parameters for HMM
    if ("m" in params) and ("p" in params):
        last_log_mu = init_log_mu
        last_p_binom = init_p_binom
    elif "m" in params:
        last_log_mu = init_log_mu
        last_p_binom = None
    elif "p" in params:
        last_log_mu = None
        last_p_binom = init_p_binom

    last_alphas = init_alphas
    last_taus = init_taus

    last_assignment = np.zeros(single_X.shape[2], dtype=int)

    for c, idx in enumerate(initial_clone_index):
        last_assignment[idx] = c

    # HMM
    for r in range(max_iter_outer):
        sample_length = np.ones(X.shape[2], dtype=int) * X.shape[0]
        remain_kwargs = {"sample_length": sample_length, "lambd": lambd}

        if f"round{r-1}_log_gamma" in allres:
            remain_kwargs["log_gamma"] = allres[f"round{r-1}_log_gamma"]

        res = pipeline_baum_welch(
            None,
            np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
                -1, 2, 1
            ),
            np.tile(lengths, X.shape[2]),
            n_states,
            base_nb_mean.flatten("F").reshape(-1, 1),
            total_bb_RD.flatten("F").reshape(-1, 1),
            np.tile(log_sitewise_transmat, X.shape[2]),
            np.repeat(tumor_prop, X.shape[0]).reshape(-1, 1),
            hmmclass=hmmclass,
            params=params,
            t=t,
            random_state=random_state,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            is_diag=is_diag,
            init_log_mu=last_log_mu,
            init_p_binom=last_p_binom,
            init_alphas=last_alphas,
            init_taus=last_taus,
            max_iter=max_iter,
            tol=tol,
            **remain_kwargs,
        )

        pred = np.argmax(res["log_gamma"], axis=0)

        (
            new_assignment,
            single_llf,
            total_llf,
        ) = hmrfmix_reassignment_posterior_concatenate(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            single_tumor_prop,
            res,
            smooth_mat,
            adjacency_mat,
            last_assignment,
            sample_ids,
            log_persample_weights,
            spatial_weight=spatial_weight,
            hmmclass=hmmclass,
        )

        # handle the case when one clone has zero spots
        if len(np.unique(new_assignment)) < X.shape[2]:
            res["assignment_before_reindex"] = new_assignment
            remaining_clones = np.sort(np.unique(new_assignment))
            re_indexing = {c: i for i, c in enumerate(remaining_clones)}
            new_assignment = np.array([re_indexing[x] for x in new_assignment])
            concat_idx = np.concatenate(
                [np.arange(c * n_obs, c * n_obs + n_obs) for c in remaining_clones]
            )
            res["log_gamma"] = res["log_gamma"][:, concat_idx]
            res["pred_cnv"] = res["pred_cnv"][concat_idx]

        # add to results
        res["prev_assignment"] = last_assignment
        res["new_assignment"] = new_assignment
        res["total_llf"] = total_llf

        # append to allres
        for k, v in res.items():
            if k == "prev_assignment":
                allres[f"round{r-1}_assignment"] = v
            elif k == "new_assignment":
                allres[f"round{r}_assignment"] = v
            else:
                allres[f"round{r}_{k}"] = v

        allres["num_iterations"] = r + 1

        np.savez(f"{outdir}/{prefix}_nstates{n_states}_{params}.npz", **allres)

        # regroup to pseudobulk
        clone_index = [
            np.where(res["new_assignment"] == c)[0]
            for c in np.sort(np.unique(res["new_assignment"]))
        ]

        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            clone_index,
            single_tumor_prop,
            threshold=tumorprop_threshold,
        )

        if "mp" in params:
            print(
                "outer iteration {}: difference between parameters = {}, {}".format(
                    r,
                    np.mean(np.abs(last_log_mu - res["new_log_mu"])),
                    np.mean(np.abs(last_p_binom - res["new_p_binom"])),
                )
            )
        elif "m" in params:
            print(
                "outer iteration {}: difference between NB parameters = {}".format(
                    r, np.mean(np.abs(last_log_mu - res["new_log_mu"]))
                )
            )
        elif "p" in params:
            print(
                "outer iteration {}: difference between BetaBinom parameters = {}".format(
                    r, np.mean(np.abs(last_p_binom - res["new_p_binom"]))
                )
            )

        print(
            "outer iteration {}: ARI between assignment = {}".format(
                r, adjusted_rand_score(last_assignment, res["new_assignment"])
            )
        )

        if (
            adjusted_rand_score(last_assignment, res["new_assignment"]) > 0.99
            or len(np.unique(res["new_assignment"])) == 1
        ):
            break

        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]
        log_persample_weights = np.ones((X.shape[2], n_samples)) * (-np.log(X.shape[2]))

        for sidx in range(n_samples):
            index = np.where(sample_ids == sidx)[0]
            this_persample_weight = np.bincount(
                res["new_assignment"][index], minlength=X.shape[2]
            ) / len(index)
            log_persample_weights[:, sidx] = np.where(
                this_persample_weight > 0, np.log(this_persample_weight), -50
            )
            log_persample_weights[:, sidx] = log_persample_weights[
                :, sidx
            ] - scipy.special.logsumexp(log_persample_weights[:, sidx])
