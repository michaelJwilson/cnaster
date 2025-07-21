import logging
import numpy as np
import networkx as nx
import copy
import scipy

from cnaster.hmm_sitewise import hmm_sitewise

logger = logging.getLogger(__name__)


def eval_neymanpearson_bafonly(
    log_emission_baf_c1, pred_c1, log_emission_baf_c2, pred_c2, bidx, n_states, res, p
):
    assert (
        log_emission_baf_c1.shape[0] == n_states
        or log_emission_baf_c1.shape[0] == 2 * n_states
    )
    # NB likelihood under the corresponding state
    llf_original = np.append(
        log_emission_baf_c1[pred_c1[bidx], bidx],
        log_emission_baf_c2[pred_c2[bidx], bidx],
    ).reshape(-1, 1)
    # NB likelihood under the switched state
    if log_emission_baf_c1.shape[0] == 2 * n_states:
        if (res["new_p_binom"][p[0], 0] > 0.5) == (res["new_p_binom"][p[1], 0] > 0.5):
            switch_pred_c1 = n_states * (pred_c1 >= n_states) + (pred_c2 % n_states)
            switch_pred_c2 = n_states * (pred_c2 >= n_states) + (pred_c1 % n_states)
        else:
            switch_pred_c1 = n_states * (pred_c1 < n_states) + (pred_c2 % n_states)
            switch_pred_c2 = n_states * (pred_c2 < n_states) + (pred_c1 % n_states)
    else:
        switch_pred_c1 = pred_c2
        switch_pred_c2 = pred_c1
    llf_switch = np.append(
        log_emission_baf_c1[switch_pred_c1[bidx], bidx],
        log_emission_baf_c2[switch_pred_c2[bidx], bidx],
    ).reshape(-1, 1)
    # NB log likelihood difference
    return np.mean(llf_original) - np.mean(llf_switch)


def eval_neymanpearson_rdrbaf(
    log_emission_rdr_c1,
    log_emission_baf_c1,
    pred_c1,
    log_emission_rdr_c2,
    log_emission_baf_c2,
    pred_c2,
    bidx,
    n_states,
    res,
    p,
):
    assert (
        log_emission_baf_c1.shape[0] == n_states
        or log_emission_baf_c1.shape[0] == 2 * n_states
    )
    # NB likelihood under the corresponding state
    llf_original = np.append(
        log_emission_rdr_c1[pred_c1[bidx], bidx]
        + log_emission_baf_c1[pred_c1[bidx], bidx],
        log_emission_rdr_c2[pred_c2[bidx], bidx]
        + log_emission_baf_c2[pred_c2[bidx], bidx],
    ).reshape(-1, 1)
    # NB likelihood under the switched state
    if log_emission_baf_c1.shape[0] == 2 * n_states:
        if (res["new_p_binom"][p[0], 0] > 0.5) == (res["new_p_binom"][p[1], 0] > 0.5):
            switch_pred_c1 = n_states * (pred_c1 >= n_states) + (pred_c2 % n_states)
            switch_pred_c2 = n_states * (pred_c2 >= n_states) + (pred_c1 % n_states)
        else:
            switch_pred_c1 = n_states * (pred_c1 < n_states) + (pred_c2 % n_states)
            switch_pred_c2 = n_states * (pred_c2 < n_states) + (pred_c1 % n_states)
    else:
        switch_pred_c1 = pred_c2
        switch_pred_c2 = pred_c1
    llf_switch = np.append(
        log_emission_rdr_c1[switch_pred_c1[bidx], bidx]
        + log_emission_baf_c1[switch_pred_c1[bidx], bidx],
        log_emission_rdr_c2[switch_pred_c2[bidx], bidx]
        + log_emission_baf_c2[switch_pred_c2[bidx], bidx],
    ).reshape(-1, 1)
    # NB log likelihood difference
    return np.mean(llf_original) - np.mean(llf_switch)


def neyman_pearson_similarity(
    X,
    base_nb_mean,
    total_bb_RD,
    res,
    threshold=2.0,
    minlength=10,
    topk=10,
    params="smp",
    tumor_prop=None,
    hmmclass=hmm_sitewise,
    **kwargs,
):
    logger.info("Solving for Neyman-Pearson similiarity.")
    
    n_obs = X.shape[0]
    n_clones = X.shape[2]
    n_states = res["new_p_binom"].shape[0]

    G = nx.Graph()
    G.add_nodes_from(np.arange(n_clones))

    lambd = np.sum(base_nb_mean, axis=1) / np.sum(base_nb_mean)

    if tumor_prop is None:
        log_emission_rdr, log_emission_baf = (
            hmmclass.compute_emission_probability_nb_betabinom(
                np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
                    -1, 2, 1
                ),
                base_nb_mean.flatten("F").reshape(-1, 1),
                res["new_log_mu"],
                res["new_alphas"],
                total_bb_RD.flatten("F").reshape(-1, 1),
                res["new_p_binom"],
                res["new_taus"],
            )
        )
    else:
        if "m" in params:
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
                        res["new_log_mu"][this_pred_cnv, :]
                        + np.log(lambd).reshape(-1, 1),
                        axis=0,
                    )
                )
            logmu_shift = np.vstack(logmu_shift)
            log_emission_rdr, log_emission_baf = (
                hmmclass.compute_emission_probability_nb_betabinom_mix(
                    np.vstack(
                        [X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]
                    ).T.reshape(-1, 2, 1),
                    base_nb_mean.flatten("F").reshape(-1, 1),
                    res["new_log_mu"],
                    res["new_alphas"],
                    total_bb_RD.flatten("F").reshape(-1, 1),
                    res["new_p_binom"],
                    res["new_taus"],
                    tumor_prop,
                    logmu_shift=logmu_shift,
                    sample_length=np.ones(n_clones, dtype=int) * n_obs,
                )
            )
        else:
            log_emission_rdr, log_emission_baf = (
                hmmclass.compute_emission_probability_nb_betabinom_mix(
                    np.vstack(
                        [X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]
                    ).T.reshape(-1, 2, 1),
                    base_nb_mean.flatten("F").reshape(-1, 1),
                    res["new_log_mu"],
                    res["new_alphas"],
                    total_bb_RD.flatten("F").reshape(-1, 1),
                    res["new_p_binom"],
                    res["new_taus"],
                    tumor_prop,
                )
            )

    log_emission_rdr = log_emission_rdr.reshape(
        (log_emission_rdr.shape[0], n_obs, n_clones), order="F"
    )

    log_emission_baf = log_emission_baf.reshape(
        (log_emission_baf.shape[0], n_obs, n_clones), order="F"
    )
    reshaped_pred = np.argmax(res["log_gamma"], axis=0).reshape((X.shape[2], -1))
    reshaped_pred_cnv = reshaped_pred % n_states
    all_test_statistics = []
    for c1 in range(n_clones):
        for c2 in range(c1 + 1, n_clones):
            unique_pair_states = [
                x
                for x in np.unique(reshaped_pred_cnv[np.array([c1, c2]), :], axis=1).T
                if x[0] != x[1]
            ]
            list_t_neymanpearson = []
            for p in unique_pair_states:
                bidx = np.where(
                    (reshaped_pred_cnv[c1, :] == p[0])
                    & (reshaped_pred_cnv[c2, :] == p[1])
                )[0]
                if "m" in params and "p" in params:
                    t_neymanpearson = eval_neymanpearson_rdrbaf(
                        log_emission_rdr[:, :, c1],
                        log_emission_baf[:, :, c1],
                        reshaped_pred[c1, :],
                        log_emission_rdr[:, :, c2],
                        log_emission_baf[:, :, c2],
                        reshaped_pred[c2, :],
                        bidx,
                        n_states,
                        res,
                        p,
                    )
                elif "p" in params:
                    t_neymanpearson = eval_neymanpearson_bafonly(
                        log_emission_baf[:, :, c1],
                        reshaped_pred[c1, :],
                        log_emission_baf[:, :, c2],
                        reshaped_pred[c2, :],
                        bidx,
                        n_states,
                        res,
                        p,
                    )
                logger.info(f"Evaluated Neyman-Pearson for {c1},{c2} with p={p} and t={t_neymanpearson:+.4f}")
                all_test_statistics.append([c1, c2, p, t_neymanpearson])
                if len(bidx) >= minlength:
                    list_t_neymanpearson.append(t_neymanpearson)
            if (
                len(list_t_neymanpearson) == 0
                or np.max(list_t_neymanpearson) < threshold
            ):
                max_v = (
                    np.max(list_t_neymanpearson)
                    if len(list_t_neymanpearson) > 0
                    else 1e-3
                )
                G.add_weighted_edges_from([(c1, c2, max_v)])
    # NB maximal cliques
    cliques = []
    for x in nx.find_cliques(G):
        this_len = len(x)
        this_weights = (
            np.sum([G.get_edge_data(a, b)["weight"] for a in x for b in x if a != b])
            / 2
        )
        cliques.append((x, this_len, this_weights))
    cliques.sort(key=lambda x: (-x[1], x[2]))
    covered_nodes = set()
    merging_groups = []
    for c in cliques:
        if len(set(c[0]) & covered_nodes) == 0:
            merging_groups.append(list(c[0]))
            covered_nodes = covered_nodes | set(c[0])
    for c in range(n_clones):
        if not (c in covered_nodes):
            merging_groups.append([c])
            covered_nodes.add(c)
    merging_groups.sort(key=lambda x: np.min(x))
    # NB clone assignment after merging
    map_clone_id = {}
    for i, x in enumerate(merging_groups):
        for z in x:
            map_clone_id[z] = i
    new_assignment = np.array([map_clone_id[x] for x in res["new_assignment"]])
    merged_res = copy.copy(res)
    merged_res["new_assignment"] = new_assignment
    merged_res["total_llf"] = np.NAN
    merged_res["pred_cnv"] = np.concatenate(
        [
            res["pred_cnv"][(c[0] * n_obs) : (c[0] * n_obs + n_obs)]
            for c in merging_groups
        ]
    )
    merged_res["log_gamma"] = np.hstack(
        [
            res["log_gamma"][:, (c[0] * n_obs) : (c[0] * n_obs + n_obs)]
            for c in merging_groups
        ]
    )

    # TODO
    logger.info(f"BAF clone merging after comparing similarity: {merging_groups}")
    
    return merging_groups, merged_res
