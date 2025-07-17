import logging

import numpy as np
import scipy.special
from cnaster.hmm_sitewise import hmm_sitewise
from cnaster.hmrf_reassignment import (
    aggr_hmrfmix_reassignment_concatenate,
    hmrfmix_reassignment_posterior_concatenate,
)
from cnaster.omics import (
    initialization_by_gmm,
    merge_pseudobulk_by_index_mix,
    pipeline_baum_welch,
)
from cnaster.spatial import compute_adjacency_mat

# Set up logger
logger = logging.getLogger(__name__)


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
    # pseudobulk
    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
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
        # assuming file f"{outdir}/{prefix}_nstates{n_states}_{params}.npz" exists. When r == 0, f"{outdir}/{prefix}_nstates{n_states}_{params}.npz" should contain two keys: "num_iterations" and f"round_-1_assignment" for clone initialization
        allres = np.load(
            f"{outdir}/{prefix}_nstates{n_states}_{params}.npz", allow_pickle=True
        )
        allres = dict(allres)
        if allres["num_iterations"] > r:
            res = {
                "new_log_mu": allres[f"round{r}_new_log_mu"],
                "new_alphas": allres[f"round{r}_new_alphas"],
                "new_p_binom": allres[f"round{r}_new_p_binom"],
                "new_taus": allres[f"round{r}_new_taus"],
                "new_log_startprob": allres[f"round{r}_new_log_startprob"],
                "new_log_transmat": allres[f"round{r}_new_log_transmat"],
                "log_gamma": allres[f"round{r}_log_gamma"],
                "pred_cnv": allres[f"round{r}_pred_cnv"],
                "llf": allres[f"round{r}_llf"],
                "total_llf": allres[f"round{r}_total_llf"],
                "prev_assignment": allres[f"round{r - 1}_assignment"],
                "new_assignment": allres[f"round{r}_assignment"],
            }
        else:
            sample_length = np.ones(X.shape[2], dtype=int) * X.shape[0]
            remain_kwargs = {"sample_length": sample_length, "lambd": lambd}
            if f"round{r - 1}_log_gamma" in allres:
                remain_kwargs["log_gamma"] = allres[f"round{r - 1}_log_gamma"]
            res = pipeline_baum_welch(
                None,
                np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
                    -1, 2, 1
                ),
                np.tile(lengths, X.shape[2]),
                n_states,  # base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), tumor_prop, \
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
            # clone assignmment
            if nodepotential == "max":
                new_assignment, single_llf, total_llf = (
                    aggr_hmrfmix_reassignment_concatenate(
                        single_X,
                        single_base_nb_mean,
                        single_total_bb_RD,
                        single_tumor_prop,
                        res,
                        pred,
                        smooth_mat,
                        adjacency_mat,
                        last_assignment,
                        sample_ids,
                        log_persample_weights,
                        spatial_weight=spatial_weight,
                        hmmclass=hmmclass,
                    )
                )
            elif nodepotential == "weighted_sum":
                new_assignment, single_llf, total_llf = (
                    hmrfmix_reassignment_posterior_concatenate(
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
                )
            else:
                raise Exception("Unknown mode for nodepotential!")
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
                    allres[f"round{r - 1}_assignment"] = v
                elif k == "new_assignment":
                    allres[f"round{r}_assignment"] = v
                else:
                    allres[f"round{r}_{k}"] = v
            allres["num_iterations"] = r + 1
            np.savez(f"{outdir}/{prefix}_nstates{n_states}_{params}.npz", **allres)
        #
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
        #
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
        # if np.all( last_assignment == res["new_assignment"] ):
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
