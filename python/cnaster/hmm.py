import copy
import logging

import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats
from numba import njit
from sklearn.mixture import GaussianMixture
from statsmodels.base.model import GenericLikelihoodModel

logger = logging.getLogger(__name__)


def initialization_by_gmm(
    n_states,
    X,
    base_nb_mean,
    total_bb_RD,
    params,
    random_state=None,
    in_log_space=True,
    only_minor=True,
    min_binom_prob=0.1,
    max_binom_prob=0.9,
):
    X_gmm_rdr, X_gmm_baf = None, None

    if "m" in params:
        if in_log_space:
            X_gmm_rdr = np.vstack(
                [np.log(X[:, 0, s] / base_nb_mean[:, s]) for s in range(X.shape[2])]
            ).T
            offset = np.mean(X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))])
            normalizetomax1 = np.max(
                X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))]
            ) - np.min(X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))])
            X_gmm_rdr = (X_gmm_rdr - offset) / normalizetomax1
        else:
            X_gmm_rdr = np.vstack(
                [X[:, 0, s] / base_nb_mean[:, s] for s in range(X.shape[2])]
            ).T
            offset = 0
            normalizetomax1 = np.max(
                X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))]
            )
            X_gmm_rdr = (X_gmm_rdr - offset) / normalizetomax1
    if "p" in params:
        X_gmm_baf = np.vstack(
            [X[:, 1, s] / total_bb_RD[:, s] for s in range(X.shape[2])]
        ).T
        X_gmm_baf[X_gmm_baf < min_binom_prob] = min_binom_prob
        X_gmm_baf[X_gmm_baf > max_binom_prob] = max_binom_prob
    # combine RDR and BAF
    if ("m" in params) and ("p" in params):
        X_gmm = np.hstack([X_gmm_rdr, X_gmm_baf])
    elif "m" in params:
        X_gmm = X_gmm_rdr
    elif "p" in params:
        X_gmm = X_gmm_baf

    # deal with NAN
    for k in range(X_gmm.shape[1]):
        last_idx_notna = -1
        for i in range(X_gmm.shape[0]):
            if last_idx_notna >= 0 and np.isnan(X_gmm[i, k]):
                X_gmm[i, k] = X_gmm[last_idx_notna, k]
            elif not np.isnan(X_gmm[i, k]):
                last_idx_notna = i

    X_gmm = X_gmm[np.sum(np.isnan(X_gmm), axis=1) == 0, :]

    # run GMM
    if random_state is None:
        gmm = GaussianMixture(n_components=n_states, max_iter=1).fit(X_gmm)
    else:
        gmm = GaussianMixture(
            n_components=n_states, max_iter=1, random_state=random_state
        ).fit(X_gmm)

    # turn gmm fitted parameters to HMM log_mu and p_binom parameters
    if ("m" in params) and ("p" in params):
        gmm_log_mu = (
            gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset
            if in_log_space
            else np.log(gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset)
        )
        gmm_p_binom = gmm.means_[:, X.shape[2] :]
        if only_minor:
            gmm_p_binom = np.where(gmm_p_binom > 0.5, 1 - gmm_p_binom, gmm_p_binom)
    elif "m" in params:
        gmm_log_mu = (
            gmm.means_ * normalizetomax1 + offset
            if in_log_space
            else np.log(gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset)
        )
        gmm_p_binom = None
    elif "p" in params:
        gmm_log_mu = None
        gmm_p_binom = gmm.means_

        if only_minor:
            gmm_p_binom = np.where(gmm_p_binom > 0.5, 1.0 - gmm_p_binom, gmm_p_binom)

    return gmm_log_mu, gmm_p_binom


@njit
def update_startprob_sitewise(lengths, log_gamma):
    """
    Input
        lengths: sum of lengths = n_observations.
        log_gamma: size 2 * n_states * n_observations. gamma[i,t] = P(q_t = i | O, lambda).
    Output
        log_startprob: n_states. Start probability after log transformation.
    """
    n_states = int(log_gamma.shape[0] / 2)
    n_obs = log_gamma.shape[1]

    assert (
        np.sum(lengths) == n_obs
    ), "Sum of lengths must be equal to the second dimension of log_gamma!"

    # indices of the start of sequences, given that the length of each sequence is in lengths
    cumlen = 0
    indices_start = []
    for le in lengths:
        indices_start.append(cumlen)
        cumlen += le
    indices_start = np.array(indices_start)

    log_startprob = np.zeros(n_states)

    # compute log_startprob of 2 * n_states
    log_startprob = mylogsumexp_ax_keep(log_gamma[:, indices_start], axis=1)

    # merge (CNV state, phase A) and (CNV state, phase B)
    log_startprob = log_startprob.flatten().reshape(2, -1)
    log_startprob = mylogsumexp_ax_keep(log_startprob, axis=0)

    # normalize such that startprob sums to 1
    log_startprob -= mylogsumexp(log_startprob)

    return log_startprob


def update_transition_sitewise(log_xi, is_diag=False):
    """
    Input
        log_xi: size (2*n_states) * (2*n_states) * n_observations. xi[i,j,t] = P(q_t=i, q_{t+1}=j | O, lambda)
    Output
        log_transmat: n_states * n_states. Transition probability after log transformation.
    """
    n_states = int(log_xi.shape[0] / 2)
    n_obs = log_xi.shape[2]

    # initialize log_transmat
    log_transmat = np.zeros((n_states, n_states))

    for i in np.arange(n_states):
        for j in np.arange(n_states):
            log_transmat[i, j] = scipy.special.logsumexp(
                np.concatenate(
                    [
                        log_xi[i, j, :],
                        log_xi[i + n_states, j, :],
                        log_xi[i, j + n_states, :],
                        log_xi[i + n_states, j + n_states, :],
                    ]
                )
            )

    # row normalize log_transmat
    if not is_diag:
        for i in np.arange(n_states):
            rowsum = scipy.special.logsumexp(log_transmat[i, :])
            log_transmat[i, :] -= rowsum
    else:
        diagsum = scipy.special.logsumexp(np.diag(log_transmat))
        totalsum = scipy.special.logsumexp(log_transmat)
        t = diagsum - totalsum
        rest = np.log((1 - np.exp(t)) / (n_states - 1))
        log_transmat = np.ones(log_transmat.shape) * rest
        np.fill_diagonal(log_transmat, t)

    return log_transmat


def update_emission_params_nb_sitewise_uniqvalues_mix(
    unique_values,
    mapping_matrices,
    log_gamma,
    base_nb_mean,
    alphas,
    tumor_prop,
    start_log_mu=None,
    fix_NB_dispersion=False,
    shared_NB_dispersion=False,
    min_log_rdr=-2,
    max_log_rdr=2,
):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    log_gamma : array, (2*n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.
    """
    n_spots = len(unique_values)
    n_states = int(log_gamma.shape[0] / 2)
    gamma = np.exp(log_gamma)

    # initialization
    new_log_mu = (
        copy.copy(start_log_mu)
        if not start_log_mu is None
        else np.zeros((n_states, n_spots))
    )
    new_alphas = copy.copy(alphas)

    # expression signal by NB distribution
    if fix_NB_dispersion:
        new_log_mu = np.zeros((n_states, n_spots))
        for s in range(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                model = sm.GLM(
                    unique_values[s][idx_nonzero, 0],
                    np.ones(len(idx_nonzero)).reshape(-1, 1),
                    family=sm.families.NegativeBinomial(alpha=alphas[i, s]),
                    exposure=unique_values[s][idx_nonzero, 1],
                    var_weights=tmp[i, idx_nonzero] + tmp[i + n_states, idx_nonzero],
                )
                res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                new_log_mu[i, s] = res.params[0]
                if not (start_log_mu is None):
                    res2 = model.fit(
                        disp=0,
                        maxiter=1500,
                        start_params=np.array([start_log_mu[i, s]]),
                        xtol=1e-4,
                        ftol=1e-4,
                    )
                    new_log_mu[i, s] = (
                        res.params[0]
                        if -model.loglike(res.params) < -model.loglike(res2.params)
                        else res2.params[0]
                    )
    else:
        if not shared_NB_dispersion:
            for s in range(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    this_tp = (mapping_matrices[s].T @ tumor_prop[:, s])[
                        idx_nonzero
                    ] / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ]
                    model = Weighted_NegativeBinomial_mix(
                        unique_values[s][idx_nonzero, 0],
                        np.ones(len(idx_nonzero)).reshape(-1, 1),
                        weights=tmp[i, idx_nonzero] + tmp[i + n_states, idx_nonzero],
                        exposure=unique_values[s][idx_nonzero, 1],
                        tumor_prop=this_tp,
                    )
                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_log_mu[i, s] = res.params[0]
                    new_alphas[i, s] = res.params[-1]

                    if not (start_log_mu is None):
                        res2 = model.fit(
                            disp=0,
                            maxiter=1500,
                            start_params=np.append(
                                [start_log_mu[i, s]], [alphas[i, s]]
                            ),
                            xtol=1e-4,
                            ftol=1e-4,
                        )
                        new_log_mu[i, s] = (
                            res.params[0]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[0]
                        )
                        new_alphas[i, s] = (
                            res.params[-1]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[-1]
                        )
        else:
            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            tp = []
            for s in range(n_spots):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(unique_values[s][idx_nonzero, 1], n_states)
                this_y = np.tile(unique_values[s][idx_nonzero, 0], n_states)
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                this_tp = np.tile(
                    (mapping_matrices[s].T @ tumor_prop[:, s])[idx_nonzero]
                    / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ],
                    n_states,
                )
                assert np.all(this_tp < 1 + 1e-4)
                this_weights = np.concatenate(
                    [
                        tmp[i, idx_nonzero] + tmp[i + n_states, idx_nonzero]
                        for i in range(n_states)
                    ]
                )
                this_features = np.zeros((n_states * len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    this_features[
                        (i * len(idx_nonzero)) : ((i + 1) * len(idx_nonzero)), i
                    ] = 1
                # only optimize for states where at least 1 SNP belongs to
                idx_state_posweight = np.array(
                    [
                        i
                        for i in range(this_features.shape[1])
                        if np.sum(this_weights[this_features[:, i] == 1]) >= 0.1
                    ]
                )
                idx_row_posweight = np.concatenate(
                    [np.where(this_features[:, k] == 1)[0] for k in idx_state_posweight]
                )
                y.append(this_y[idx_row_posweight])
                exposure.append(this_exposure[idx_row_posweight])
                weights.append(this_weights[idx_row_posweight])
                features.append(
                    this_features[idx_row_posweight, :][:, idx_state_posweight]
                )
                state_posweights.append(idx_state_posweight)
                tp.append(this_tp[idx_row_posweight])

            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            tp = np.concatenate(tp)
            model = Weighted_NegativeBinomial_mix(
                y,
                features,
                weights=weights,
                exposure=exposure,
                tumor_prop=tp,
                penalty=0,
            )
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            for s, idx_state_posweight in enumerate(state_posweights):
                l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                new_log_mu[idx_state_posweight, s] = res.params[l1:l2]
            if res.params[-1] > 0:
                new_alphas[:, :] = res.params[-1]
            if not (start_log_mu is None):
                res2 = model.fit(
                    disp=0,
                    maxiter=1500,
                    start_params=np.concatenate(
                        [
                            start_log_mu[idx_state_posweight, s]
                            for s, idx_state_posweight in enumerate(state_posweights)
                        ]
                        + [np.ones(1) * alphas[0, s]]
                    ),
                    xtol=1e-4,
                    ftol=1e-4,
                )
                if model.nloglikeobs(res2.params) < model.nloglikeobs(res.params):
                    for s, idx_state_posweight in enumerate(state_posweights):
                        l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                        l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                        new_log_mu[idx_state_posweight, s] = res2.params[l1:l2]
                    if res2.params[-1] > 0:
                        new_alphas[:, :] = res2.params[-1]
    new_log_mu[new_log_mu > max_log_rdr] = max_log_rdr
    new_log_mu[new_log_mu < min_log_rdr] = min_log_rdr
    return new_log_mu, new_alphas


def update_emission_params_bb_sitewise_uniqvalues_mix(
    unique_values,
    mapping_matrices,
    log_gamma,
    total_bb_RD,
    taus,
    tumor_prop,
    start_p_binom=None,
    fix_BB_dispersion=False,
    shared_BB_dispersion=False,
    percent_threshold=0.99,
    min_binom_prob=0.01,
    max_binom_prob=0.99,
):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    log_gamma : array, (2*n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.
    """
    n_spots = len(unique_values)
    n_states = int(log_gamma.shape[0] / 2)
    gamma = np.exp(log_gamma)

    # initialization
    new_p_binom = (
        copy.copy(start_p_binom)
        if not start_p_binom is None
        else np.ones((n_states, n_spots)) * 0.5
    )
    new_taus = copy.copy(taus)

    if fix_BB_dispersion:
        for s in np.arange(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                if (
                    np.sum(tmp[i, idx_nonzero]) + np.sum(tmp[i + n_states, idx_nonzero])
                    >= 0.1
                ):
                    this_tp = (mapping_matrices[s].T @ tumor_prop[:, s])[
                        idx_nonzero
                    ] / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ]
                    assert np.all(this_tp < 1 + 1e-4)
                    model = Weighted_BetaBinom_fixdispersion_mix(
                        np.append(
                            unique_values[s][idx_nonzero, 0],
                            unique_values[s][idx_nonzero, 1]
                            - unique_values[s][idx_nonzero, 0],
                        ),
                        np.ones(2 * len(idx_nonzero)).reshape(-1, 1),
                        taus[i, s],
                        weights=np.append(
                            tmp[i, idx_nonzero], tmp[i + n_states, idx_nonzero]
                        ),
                        exposure=np.append(
                            unique_values[s][idx_nonzero, 1],
                            unique_values[s][idx_nonzero, 1],
                        ),
                        tumor_prop=this_tp,
                    )

                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_p_binom[i, s] = res.params[0]
                    if not (start_p_binom is None):
                        res2 = model.fit(
                            disp=0,
                            maxiter=1500,
                            start_params=np.array(start_p_binom[i, s]),
                            xtol=1e-4,
                            ftol=1e-4,
                        )
                        new_p_binom[i, s] = (
                            res.params[0]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[0]
                        )
    else:
        if not shared_BB_dispersion:
            for s in np.arange(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                    if (
                        np.sum(tmp[i, idx_nonzero])
                        + np.sum(tmp[i + n_states, idx_nonzero])
                        >= 0.1
                    ):
                        this_tp = (mapping_matrices[s].T @ tumor_prop[:, s])[
                            idx_nonzero
                        ] / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                            idx_nonzero
                        ]
                        assert np.all(this_tp < 1 + 1e-4)
                        model = Weighted_BetaBinom_mix(
                            np.append(
                                unique_values[s][idx_nonzero, 0],
                                unique_values[s][idx_nonzero, 1]
                                - unique_values[s][idx_nonzero, 0],
                            ),
                            np.ones(2 * len(idx_nonzero)).reshape(-1, 1),
                            weights=np.append(
                                tmp[i, idx_nonzero], tmp[i + n_states, idx_nonzero]
                            ),
                            exposure=np.append(
                                unique_values[s][idx_nonzero, 1],
                                unique_values[s][idx_nonzero, 1],
                            ),
                            tumor_prop=this_tp,
                        )

                        res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                        new_p_binom[i, s] = res.params[0]
                        new_taus[i, s] = res.params[-1]
                        if not (start_p_binom is None):
                            res2 = model.fit(
                                disp=0,
                                maxiter=1500,
                                start_params=np.append(
                                    [start_p_binom[i, s]], [taus[i, s]]
                                ),
                                xtol=1e-4,
                                ftol=1e-4,
                            )
                            new_p_binom[i, s] = (
                                res.params[0]
                                if model.nloglikeobs(res.params)
                                < model.nloglikeobs(res2.params)
                                else res2.params[0]
                            )
                            new_taus[i, s] = (
                                res.params[-1]
                                if model.nloglikeobs(res.params)
                                < model.nloglikeobs(res2.params)
                                else res2.params[-1]
                            )
        else:
            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            tp = []
            for s in np.arange(n_spots):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(
                    np.append(
                        unique_values[s][idx_nonzero, 1],
                        unique_values[s][idx_nonzero, 1],
                    ),
                    n_states,
                )
                this_y = np.tile(
                    np.append(
                        unique_values[s][idx_nonzero, 0],
                        unique_values[s][idx_nonzero, 1]
                        - unique_values[s][idx_nonzero, 0],
                    ),
                    n_states,
                )
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                this_tp = np.tile(
                    (mapping_matrices[s].T @ tumor_prop[:, s])[idx_nonzero]
                    / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ],
                    n_states,
                )
                assert np.all(this_tp < 1 + 1e-4)
                this_weights = np.concatenate(
                    [
                        np.append(tmp[i, idx_nonzero], tmp[i + n_states, idx_nonzero])
                        for i in range(n_states)
                    ]
                )
                this_features = np.zeros((2 * n_states * len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    this_features[
                        (i * 2 * len(idx_nonzero)) : ((i + 1) * 2 * len(idx_nonzero)), i
                    ] = 1
                # only optimize for states where at least 1 SNP belongs to
                idx_state_posweight = np.array(
                    [
                        i
                        for i in range(this_features.shape[1])
                        if np.sum(this_weights[this_features[:, i] == 1]) >= 0.1
                    ]
                )
                idx_row_posweight = np.concatenate(
                    [np.where(this_features[:, k] == 1)[0] for k in idx_state_posweight]
                )
                y.append(this_y[idx_row_posweight])
                exposure.append(this_exposure[idx_row_posweight])
                weights.append(this_weights[idx_row_posweight])
                features.append(
                    this_features[idx_row_posweight, :][:, idx_state_posweight]
                )
                state_posweights.append(idx_state_posweight)
                tp.append(this_tp[idx_row_posweight])

            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            tp = np.concatenate(tp)
            model = Weighted_BetaBinom_mix(
                y, features, weights=weights, exposure=exposure, tumor_prop=tp
            )
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            for s, idx_state_posweight in enumerate(state_posweights):
                l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                new_p_binom[idx_state_posweight, s] = res.params[l1:l2]
            if res.params[-1] > 0:
                new_taus[:, :] = res.params[-1]
            if not (start_p_binom is None):
                res2 = model.fit(
                    disp=0,
                    maxiter=1500,
                    start_params=np.concatenate(
                        [
                            start_p_binom[idx_state_posweight, s]
                            for s, idx_state_posweight in enumerate(state_posweights)
                        ]
                        + [np.ones(1) * taus[0, s]]
                    ),
                    xtol=1e-4,
                    ftol=1e-4,
                )
                if model.nloglikeobs(res2.params) < model.nloglikeobs(res.params):
                    for s, idx_state_posweight in enumerate(state_posweights):
                        l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                        l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                        new_p_binom[idx_state_posweight, s] = res2.params[l1:l2]
                    if res2.params[-1] > 0:
                        new_taus[:, :] = res2.params[-1]
    new_p_binom[new_p_binom < min_binom_prob] = min_binom_prob
    new_p_binom[new_p_binom > max_binom_prob] = max_binom_prob
    return new_p_binom, new_taus


def compute_posterior_obs(log_alpha, log_beta):
    """
    Input
        log_alpha: output from forward_lattice_gaussian. size n_states * n_observations. alpha[j, t] = P(o_1, ... o_t, q_t = j | lambda).
        log_beta: output from backward_lattice_gaussian. size n_states * n_observations. beta[i, t] = P(o_{t+1}, ..., o_T | q_t = i, lambda).
    Output:
        log_gamma: size n_states * n_observations. gamma[i,t] = P(q_t = i | O, lambda). gamma[i, t] propto alpha[i,t] * beta[i,t]
    """
    n_states = log_alpha.shape[0]
    n_obs = log_alpha.shape[1]

    log_gamma = np.zeros((n_states, n_obs))
    log_gamma = log_alpha + log_beta

    if np.any(np.sum(log_gamma, axis=0) == 0):
        logger.error("Sum of posterior probability is zero for some observations!")
        raise RuntimeError()

    log_gamma -= scipy.special.logsumexp(log_gamma, axis=0)

    return log_gamma


@njit
def compute_posterior_transition_sitewise(
    log_alpha, log_beta, log_transmat, log_emission
):
    """
    Input
        log_alpha: output from forward_lattice_gaussian. size n_states * n_observations. alpha[j, t] = P(o_1, ... o_t, q_t = j | lambda).
        log_beta: output from backward_lattice_gaussian. size n_states * n_observations. beta[i, t] = P(o_{t+1}, ..., o_T | q_t = i, lambda).
        log_transmat: n_states * n_states. Transition probability after log transformation.
        log_emission: n_states * n_observations * n_spots. Log probability.
    Output:
        log_xi: size n_states * n_states * (n_observations-1). xi[i,j,t] = P(q_t=i, q_{t+1}=j | O, lambda)
    """
    n_states = int(log_alpha.shape[0] / 2)
    n_obs = log_alpha.shape[1]

    log_xi = np.zeros((2 * n_states, 2 * n_states, n_obs - 1))

    for i in np.arange(2 * n_states):
        for j in np.arange(2 * n_states):
            for t in np.arange(n_obs - 1):
                # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities.
                # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
                log_xi[i, j, t] = (
                    log_alpha[i, t]
                    + log_transmat[
                        i - n_states * int(i / n_states),
                        j - n_states * int(j / n_states),
                    ]
                    + np.sum(log_emission[j, t + 1, :])
                    + log_beta[j, t + 1]
                )

    for t in np.arange(n_obs - 1):
        log_xi[:, :, t] -= mylogsumexp(log_xi[:, :, t])

    return log_xi


def pipeline_baum_welch(
    output_prefix,
    X,
    lengths,
    n_states,
    base_nb_mean,
    total_bb_RD,
    log_sitewise_transmat,
    tumor_prop=None,
    hmmclass=hmm_sitewise,
    params="smp",
    t=1 - 1e-6,
    random_state=0,
    in_log_space=True,
    only_minor=False,
    fix_NB_dispersion=False,
    shared_NB_dispersion=True,
    fix_BB_dispersion=False,
    shared_BB_dispersion=True,
    init_log_mu=None,
    init_p_binom=None,
    init_alphas=None,
    init_taus=None,
    is_diag=True,
    max_iter=100,
    tol=1e-4,
    **kwargs,
):
    # initialization
    n_spots = X.shape[2]

    if ((init_log_mu is None) and ("m" in params)) or (
        (init_p_binom is None) and ("p" in params)
    ):
        tmp_log_mu, tmp_p_binom = initialization_by_gmm(
            n_states,
            X,
            base_nb_mean,
            total_bb_RD,
            params,
            random_state=random_state,
            in_log_space=in_log_space,
            only_minor=only_minor,
        )

        if (init_log_mu is None) and ("m" in params):
            init_log_mu = tmp_log_mu
        if (init_p_binom is None) and ("p" in params):
            init_p_binom = tmp_p_binom

    logger.info(f"Initialized log_mu: {init_log_mu}")
    logger.info(f"Initialized p_binom: {init_p_binom}")

    # fit HMM-NB-BetaBinom
    hmmmodel = hmmclass(params=params, t=t)

    remain_kwargs = {
        k: v for k, v in kwargs.items() if k in ["lambd", "sample_length", "log_gamma"]
    }
    
    (
        new_log_mu,
        new_alphas,
        new_p_binom,
        new_taus,
        new_log_startprob,
        new_log_transmat,
        log_gamma,
    ) = hmmmodel.run_baum_welch_nb_bb(
        X,
        lengths,
        n_states,
        base_nb_mean,
        total_bb_RD,
        log_sitewise_transmat,
        tumor_prop,
        fix_NB_dispersion=fix_NB_dispersion,
        shared_NB_dispersion=shared_NB_dispersion,
        fix_BB_dispersion=fix_BB_dispersion,
        shared_BB_dispersion=shared_BB_dispersion,
        is_diag=is_diag,
        init_log_mu=init_log_mu,
        init_p_binom=init_p_binom,
        init_alphas=init_alphas,
        init_taus=init_taus,
        max_iter=max_iter,
        tol=tol,
        **remain_kwargs,
    )

    # likelihood
    if tumor_prop is None:
        (
            log_emission_rdr,
            log_emission_baf,
        ) = hmmclass.compute_emission_probability_nb_betabinom(
            X, base_nb_mean, new_log_mu, new_alphas, total_bb_RD, new_p_binom, new_taus
        )
        log_emission = log_emission_rdr + log_emission_baf
    else:
        if ("m" in params) and ("sample_length" in kwargs):
            logmu_shift = []
            for c in range(len(kwargs["sample_length"])):
                this_pred_cnv = (
                    np.argmax(
                        log_gamma[
                            :,
                            np.sum(kwargs["sample_length"][:c]) : np.sum(
                                kwargs["sample_length"][: (c + 1)]
                            ),
                        ],
                        axis=0,
                    )
                    % n_states
                )
                logmu_shift.append(
                    scipy.special.logsumexp(
                        new_log_mu[this_pred_cnv, :]
                        + np.log(kwargs["lambd"]).reshape(-1, 1),
                        axis=0,
                    )
                )
            logmu_shift = np.vstack(logmu_shift)
            (
                log_emission_rdr,
                log_emission_baf,
            ) = hmmclass.compute_emission_probability_nb_betabinom_mix(
                X,
                base_nb_mean,
                new_log_mu,
                new_alphas,
                total_bb_RD,
                new_p_binom,
                new_taus,
                tumor_prop,
                logmu_shift=logmu_shift,
                sample_length=kwargs["sample_length"],
            )
        else:
            (
                log_emission_rdr,
                log_emission_baf,
            ) = hmmclass.compute_emission_probability_nb_betabinom_mix(
                X,
                base_nb_mean,
                new_log_mu,
                new_alphas,
                total_bb_RD,
                new_p_binom,
                new_taus,
                tumor_prop,
            )

        log_emission = log_emission_rdr + log_emission_baf

    log_alpha = hmmclass.forward_lattice(
        lengths,
        new_log_transmat,
        new_log_startprob,
        log_emission,
        log_sitewise_transmat,
    )

    llf = np.sum(scipy.special.logsumexp(log_alpha[:, np.cumsum(lengths) - 1], axis=0))

    log_beta = hmmclass.backward_lattice(
        lengths,
        new_log_transmat,
        new_log_startprob,
        log_emission,
        log_sitewise_transmat,
    )

    log_gamma = compute_posterior_obs(log_alpha, log_beta)

    pred = np.argmax(log_gamma, axis=0)
    pred_cnv = pred % n_states

    if not output_prefix is None:
        tmp = np.log10(1 - t)
        np.savez(
            f"{output_prefix}_nstates{n_states}_{params}_{tmp:.0f}_seed{random_state}.npz",
            new_log_mu=new_log_mu,
            new_alphas=new_alphas,
            new_p_binom=new_p_binom,
            new_taus=new_taus,
            new_log_startprob=new_log_startprob,
            new_log_transmat=new_log_transmat,
            log_gamma=log_gamma,
            pred_cnv=pred_cnv,
            llf=llf,
        )
    else:
        res = {
            "new_log_mu": new_log_mu,
            "new_alphas": new_alphas,
            "new_p_binom": new_p_binom,
            "new_taus": new_taus,
            "new_log_startprob": new_log_startprob,
            "new_log_transmat": new_log_transmat,
            "log_gamma": log_gamma,
            "pred_cnv": pred_cnv,
            "llf": llf,
        }

        return res