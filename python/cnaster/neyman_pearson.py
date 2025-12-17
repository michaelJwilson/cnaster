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
    # NB log likelihood under the corresponding state for segments in the copy state
    #    pair selected by bidx.
    llf_original = np.append(
        log_emission_rdr_c1[pred_c1[bidx], bidx]
        + log_emission_baf_c1[pred_c1[bidx], bidx],
        log_emission_rdr_c2[pred_c2[bidx], bidx]
        + log_emission_baf_c2[pred_c2[bidx], bidx],
    ).reshape(-1, 1)

    # NB likelihood under the switched state, p is the copy state pair.
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

    # NB mean log likelihood difference.
    return np.mean(llf_original) - np.mean(llf_switch)


def neyman_pearson_similarity(
    X,
    base_nb_mean,
    total_bb_RD,
    res,
    threshold=None,
    minlength=10,  # MAGIC
    topk=10,
    params="smp",
    tumor_prop=None,
    hmmclass=hmm_sitewise,
    **kwargs,
):
    if threshold is None:
        threshold = get_global_config().hmrf.np_threshold
    
    logger.info(
        f"Solving for Neyman-Pearson similiarity with threshold={threshold} and {hmmclass.__name__} instance with:\nnew_log_mu=\n{res['new_log_mu']}\nnew_p_binom=\n{res['new_p_binom']}"
    )

    n_obs, _, n_clones = X.shape
    n_states = res["new_p_binom"].shape[0]

    # NB one node per clone.
    G = nx.Graph()
    G.add_nodes_from(np.arange(n_clones))

    # NB normalized baseline expression.
    lambd = np.sum(base_nb_mean, axis=1) / np.sum(base_nb_mean)

    # TODO clone stack.
    if tumor_prop is None:
        (
            log_emission_rdr,
            log_emission_baf,
        ) = hmmclass.compute_emission_probability_nb_betabinom(
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

            # TODO clone stack.
            (
                log_emission_rdr,
                log_emission_baf,
            ) = hmmclass.compute_emission_probability_nb_betabinom_mix(
                np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
                    -1, 2, 1
                ),
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
        else:
            (
                log_emission_rdr,
                log_emission_baf,
            ) = hmmclass.compute_emission_probability_nb_betabinom_mix(
                np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
                    -1, 2, 1
                ),
                base_nb_mean.flatten("F").reshape(-1, 1),
                res["new_log_mu"],
                res["new_alphas"],
                total_bb_RD.flatten("F").reshape(-1, 1),
                res["new_p_binom"],
                res["new_taus"],
                tumor_prop,
            )

    log_emission_rdr = log_emission_rdr.reshape(
        (log_emission_rdr.shape[0], n_obs, n_clones), order="F"
    )

    log_emission_baf = log_emission_baf.reshape(
        (log_emission_baf.shape[0], n_obs, n_clones), order="F"
    )

    reshaped_pred = np.argmax(res["log_gamma"], axis=0).reshape((X.shape[2], -1))

    # NB MAP copy number state.
    reshaped_pred_cnv = reshaped_pred % n_states

    all_test_statistics = []

    # NB all distinct clone pairs.
    for c1 in range(n_clones):
        for c2 in range(c1 + 1, n_clones):
            # NB unique copy number state pairs, i.e. (A, B) for the copy states A,B in two clones
            #    @ same segment.
            unique_pair_states = [
                x
                for x in np.unique(reshaped_pred_cnv[np.array([c1, c2]), :], axis=1).T
                if x[0] != x[1]
            ]

            # NB Neyman-Pearson test statistics for all copy state pairs in this clone pair.
            list_t_neymanpearson = []

            for p in unique_pair_states:
                # NB rows assigned to this copy state pair.
                bidx = np.where(
                    (reshaped_pred_cnv[c1, :] == p[0])
                    & (reshaped_pred_cnv[c2, :] == p[1])
                )[0]

                # NB log likelihood difference under switching copy state between clones.
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

                logger.info(
                    f"Evaluated NP statistic={t_neymanpearson:+.4f} for clone pair ({c1},{c2}) & copy state pair p={p}"
                )

                all_test_statistics.append([c1, c2, p, t_neymanpearson])

                # NB number of genomic bins with this copy state pair across clones.
                if len(bidx) >= minlength:
                    list_t_neymanpearson.append(t_neymanpearson)
                else:
                    logger.warning(
                        f"Copy state pair fails to meet segment usage criteria ({len(bidx)}/{minlength}) and is not used for merging."
                    )

            # NB As there are no copy state pairs with sufficient usage, or the max. NP distinction between a copy state pair is less than desired,
            #    this pair is a candidate to be merged.
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

                logger.info(
                    f"Added edge for candidate clone pair {c1}-{c2} to be merged with edge weight {max_v}"
                )
            else:
                logger.warning(
                    f"Candidate clone pair found to be distinct with max_t={np.max(list_t_neymanpearson)} vs threshold={threshold}."
                )

    # NB  cliques: set of nodes that are all neighbors.
    #     maximal cliques: clique that is not a sub-set of any larger clique.
    cliques = []

    # NB  returns iterator over maximal cliques, each of which is a list of nodes in G.
    #     see https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.clique.find_cliques.html
    maximal_cliques = nx.find_cliques(G)

    # NB we sort the maximal cliques by size, breaking ties according to the sum of the max NP statisic for all pairs in the clique.
    for x in maximal_cliques:
        # NB number of nodes in clique, presumably.
        clique_size = len(x)

        # NB sum of edge weights, dropping (b,a) given (a,b).
        clique_weights = (
            np.sum([G.get_edge_data(a, b)["weight"] for a in x for b in x if a != b])
            / 2.0
        )
        cliques.append((x, clique_size, clique_weights))

    # NB -x[1]: Sorts by clique size (number of nodes) in descending order (i.e. largest cliques first).
    #     x[2]: For cliques of the same size, sorts by total edge weight in ascending order (smaller weights first),
    #           i.e.
    cliques.sort(key=lambda x: (-x[1], x[2]))

    logger.info(f"Found sorted, maximal cliques for NP merging:\n{cliques}")

    # NB all nodes assigned to a group, new clone?
    covered_nodes = set()

    # NB stores final groups of nodes (cliques or singletons).
    merging_groups = []

    for c in cliques:
        # NB if none of the nodes in this clique are already in covered_nodes,
        #    add maximal clique as a new group.
        if len(set(c[0]) & covered_nodes) == 0:
            merging_groups.append(list(c[0]))
            covered_nodes = covered_nodes | set(c[0])

    # NB add all of original clones that are not merge candidates as singletons.
    for c in range(n_clones):
        if not (c in covered_nodes):
            merging_groups.append([c])
            covered_nodes.add(c)

    # NB sorts the groups so that those with the smallest node indices come first.
    merging_groups.sort(key=lambda x: np.min(x))

    # NB new clone assignment after merging clones.
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

    logger.info(f"New merged groups  after comparing NP similarity:\n{merging_groups}")

    return merging_groups, merged_res


def compute_neymanpearson_stats(
    X, base_nb_mean, total_bb_RD, res, params, tumor_prop, hmmclass
):
    n_obs = X.shape[0]
    n_states = res["new_p_binom"].shape[0]
    n_clones = X.shape[2]
    lambd = np.sum(base_nb_mean, axis=1) / np.sum(base_nb_mean)

    if tumor_prop is None:
        (
            log_emission_rdr,
            log_emission_baf,
        ) = hmmclass.compute_emission_probability_nb_betabinom(
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
            (
                log_emission_rdr,
                log_emission_baf,
            ) = hmmclass.compute_emission_probability_nb_betabinom_mix(
                np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
                    -1, 2, 1
                ),
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
        else:
            (
                log_emission_rdr,
                log_emission_baf,
            ) = hmmclass.compute_emission_probability_nb_betabinom_mix(
                np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
                    -1, 2, 1
                ),
                base_nb_mean.flatten("F").reshape(-1, 1),
                res["new_log_mu"],
                res["new_alphas"],
                total_bb_RD.flatten("F").reshape(-1, 1),
                res["new_p_binom"],
                res["new_taus"],
                tumor_prop,
            )
    log_emission_rdr = log_emission_rdr.reshape(
        (log_emission_rdr.shape[0], n_obs, n_clones), order="F"
    )
    log_emission_baf = log_emission_baf.reshape(
        (log_emission_baf.shape[0], n_obs, n_clones), order="F"
    )
    reshaped_pred = np.argmax(res["log_gamma"], axis=0).reshape((X.shape[2], -1))
    reshaped_pred_cnv = reshaped_pred % n_states
    all_test_statistics = {
        (c1, c2): [] for c1 in range(n_clones) for c2 in range(c1 + 1, n_clones)
    }
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
                all_test_statistics[(c1, c2)].append((p[0], p[1], t_neymanpearson))

    return all_test_statistics


def combine_similar_states_across_clones(
    X,
    base_nb_mean,
    total_bb_RD,
    res,
    params="smp",
    tumor_prop=None,
    hmmclass=hmm_sitewise,
    merge_threshold=0.1,
    **kwargs,
):
    n_obs, _, n_clones = X.shape
    n_states = res["new_p_binom"].shape[0]
    reshaped_pred = np.argmax(res["log_gamma"], axis=0).reshape((X.shape[2], -1))

    # NB drop phasing information.
    reshaped_pred_cnv = reshaped_pred % n_states

    all_test_statistics = compute_neymanpearson_stats(
        X, base_nb_mean, total_bb_RD, res, params, tumor_prop, hmmclass
    )

    # NB make the (distinct) pair of states consistent between clone c1 and clone c2 if their t_neymanpearson test statistics is small
    for c1 in range(n_clones):
        for c2 in range(c1 + 1, n_clones):
            list_t_neymanpearson = all_test_statistics[(c1, c2)]

            for p1, p2, t_neymanpearson in list_t_neymanpearson:
                if t_neymanpearson < merge_threshold:
                    c_keep = (
                        c1
                        if np.sum(total_bb_RD[:, c1]) > np.sum(total_bb_RD[:, c2])
                        else c2
                    )
                    c_change = c2 if c_keep == c1 else c1
                    bidx = np.where(
                        (reshaped_pred_cnv[c1, :] == p1)
                        & (reshaped_pred_cnv[c2, :] == p2)
                    )[0]
                    res["pred_cnv"][(c_change * n_obs) : (c_change * n_obs + n_obs)][
                        bidx
                    ] = res["pred_cnv"][(c_keep * n_obs) : (c_keep * n_obs + n_obs)][
                        bidx
                    ]
                    logger.info(
                        f"Merging states {[p1,p2]} in clone {c1} and clone {c2}. NP statistics = {t_neymanpearson}"
                    )
    return res
