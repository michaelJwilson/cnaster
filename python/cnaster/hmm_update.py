import copy

import numpy as np
import scipy
from cnaster.hmm_emission import Weighted_BetaBinom
from cnaster.hmm_utils import mylogsumexp, mylogsumexp_ax_keep
from numba import njit


def update_transition_sitewise(log_xi, is_diag=False):
    """
    Input
        log_xi: size (2*n_states) * (2*n_states) * n_observations. xi[i,j,t] = P(q_t=i, q_{t+1}=j | O, lambda)
    Output
        log_transmat: n_states * n_states. Transition probability after log transformation.
    """
    logger.debug("Updating (phasing) transition matrix given xi.")

    n_states = int(log_xi.shape[0] / 2)
    n_obs = log_xi.shape[2]

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


def update_transition_nophasing(log_xi, is_diag=False):
    """
    Input
        log_xi: size (n_states) * (n_states) * n_observations. xi[i,j,t] = P(q_t=i, q_{t+1}=j | O, lambda)
    Output
        log_transmat: n_states * n_states. Transition probability after log transformation.
    """
    logger.debug("Updating (no phasing) transition matrix given xi.")

    n_states = log_xi.shape[0]
    n_obs = log_xi.shape[2]

    log_transmat = np.zeros((n_states, n_states))
    for i in np.arange(n_states):
        for j in np.arange(n_states):
            log_transmat[i, j] = scipy.special.logsumexp(log_xi[i, j, :])

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


@njit
def update_startprob_sitewise(lengths, log_gamma):
    """
    Input
        lengths: sum of lengths = n_observations.
        log_gamma: size 2 * n_states * n_observations. gamma[i,t] = P(q_t = i | O, lambda).
    Output
        log_startprob: n_states. Start probability after log transformation.
    """
    logger.debug("Updating (phasing) start probability given gamma.")

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


@njit
def update_startprob_nophasing(lengths, log_gamma):
    """
    Input
        lengths: sum of lengths = n_observations.
        log_gamma: size n_states * n_observations. gamma[i,t] = P(q_t = i | O, lambda).
    Output
        log_startprob: n_states. Start probability after loog transformation.
    """
    logger.debug("Updating (no phasing) start probability given gamma.")

    n_states = log_gamma.shape[0]
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
    # initialize log_startprob
    log_startprob = np.zeros(n_states)
    # compute log_startprob of n_states
    log_startprob = mylogsumexp_ax_keep(log_gamma[:, indices_start], axis=1)
    # normalize such that startprob sums to 1
    log_startprob -= mylogsumexp(log_startprob)
    return log_startprob


def update_emission_params_nb_sitewise_uniqvalues(
    unique_values,
    mapping_matrices,
    log_gamma,
    base_nb_mean,
    alphas,
    start_log_mu=None,
    fix_NB_dispersion=False,
    shared_NB_dispersion=False,
    min_log_rdr=-2,
    max_log_rdr=2,
    min_estep_weight=0.1,
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

    new_log_mu = (
        copy.copy(start_log_mu)
        if not start_log_mu is None
        else np.zeros((n_states, n_spots))
    )
    new_alphas = copy.copy(alphas)

    if fix_NB_dispersion:
        logger.info("Updating (phasing) NB emission parameters with fixed dispersion.")

        new_log_mu = np.zeros((n_states, n_spots))

        for s in range(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
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
            logger.info("Updating (phasing) NB emission parameters with free dispersion.")

            for s in range(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    model = Weighted_NegativeBinomial(
                        unique_values[s][idx_nonzero, 0],
                        np.ones(len(idx_nonzero)).reshape(-1, 1),
                        weights=tmp[i, idx_nonzero] + tmp[i + n_states, idx_nonzero],
                        exposure=unique_values[s][idx_nonzero, 1],
                        penalty=0,
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
            logger.info("Updating (phasing) NB emission parameters with shared dispersion.")

            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            for s in range(n_spots):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(unique_values[s][idx_nonzero, 1], n_states)
                this_y = np.tile(unique_values[s][idx_nonzero, 0], n_states)
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
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
                        if np.sum(this_weights[this_features[:, i] == 1])
                        >= min_estep_weight
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
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            model = Weighted_NegativeBinomial(
                y, features, weights=weights, exposure=exposure
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


def update_emission_params_nb_nophasing_uniqvalues(
    unique_values,
    mapping_matrices,
    log_gamma,
    alphas,
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

    log_gamma : array, (n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.
    """
    n_spots = len(unique_values)
    n_states = log_gamma.shape[0]
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
        logger.info("Updating (no phasing) NB emission parameters with fixed dispersion.")

        new_log_mu = np.zeros((n_states, n_spots))
        for s in range(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                model = sm.GLM(
                    unique_values[s][idx_nonzero, 0],
                    np.ones(len(idx_nonzero)).reshape(-1, 1),
                    family=sm.families.NegativeBinomial(alpha=alphas[i, s]),
                    exposure=unique_values[s][idx_nonzero, 1],
                    var_weights=tmp[i, idx_nonzero],
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
            logger.info("Updating (no phasing) NB emission parameters with free dispersion.")

            for s in range(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    model = Weighted_NegativeBinomial(
                        unique_values[s][idx_nonzero, 0],
                        np.ones(len(idx_nonzero)).reshape(-1, 1),
                        weights=tmp[i, idx_nonzero],
                        exposure=unique_values[s][idx_nonzero, 1],
                        penalty=0,
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
            logger.info("Updating (no phasing) NB emission parameters with shared dispersion.")

            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            for s in range(n_spots):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(unique_values[s][idx_nonzero, 1], n_states)
                this_y = np.tile(unique_values[s][idx_nonzero, 0], n_states)
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
                this_weights = np.concatenate(
                    [tmp[i, idx_nonzero] for i in range(n_states)]
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
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            model = Weighted_NegativeBinomial(
                y, features, weights=weights, exposure=exposure
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

    new_log_mu = (
        copy.copy(start_log_mu)
        if not start_log_mu is None
        else np.zeros((n_states, n_spots))
    )
    new_alphas = copy.copy(alphas)

    # expression signal by NB distribution
    if fix_NB_dispersion:
        logger.info("Updating (phasing, tumor mix) NB emission parameters with fixed dispersion.")

        new_log_mu = np.zeros((n_states, n_spots))
        for s in range(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
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
            logger.info("Updating (phasing, tumor mix) NB emission parameters with free dispersion.")

            for s in range(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
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
            logger.info("Updating (phasing, tumor mix) NB emission parameters with shared dispersion.")

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
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
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


def update_emission_params_nb_nophasing_uniqvalues_mix(
    unique_values,
    mapping_matrices,
    log_gamma,
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

    log_gamma : array, (n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.
    """
    n_spots = len(unique_values)
    n_states = log_gamma.shape[0]
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
        logger.info("Updating (no phasing, tumor mix) NB emission parameters with fixed dispersion.")

        new_log_mu = np.zeros((n_states, n_spots))
        for s in range(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                model = sm.GLM(
                    unique_values[s][idx_nonzero, 0],
                    np.ones(len(idx_nonzero)).reshape(-1, 1),
                    family=sm.families.NegativeBinomial(alpha=alphas[i, s]),
                    exposure=unique_values[s][idx_nonzero, 1],
                    var_weights=tmp[i, idx_nonzero],
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
            logger.info("Updating (no phasing, tumor mix) NB emission parameters with free dispersion.")

            for s in range(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
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
                        weights=tmp[i, idx_nonzero],
                        exposure=unique_values[s][idx_nonzero, 1],
                        tumor_prop=this_tp,
                    )
                    # tumor_prop=tumor_prop[s], penalty=0)
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
            logger.info("Updating (no phasing, tumor mix) NB emission parameters with shared dispersion.")

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
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
                this_tp = np.tile(
                    (mapping_matrices[s].T @ tumor_prop[:, s])[idx_nonzero]
                    / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ],
                    n_states,
                )
                assert np.all(this_tp < 1 + 1e-4)
                this_weights = np.concatenate(
                    [tmp[i, idx_nonzero] for i in range(n_states)]
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


def update_emission_params_bb_sitewise_uniqvalues(
    unique_values,
    mapping_matrices,
    log_gamma,
    total_bb_RD,
    taus,
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
        logger.info("Updating (phasing) BAF emission parameters with fixed dispersion.")

        for s in np.arange(len(unique_values)):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                if (
                    np.sum(tmp[i, idx_nonzero]) + np.sum(tmp[i + n_states, idx_nonzero])
                    >= 0.1
                ):
                    model = Weighted_BetaBinom_fixdispersion(
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
            logger.info("Updating (phasing) BAF emission parameters with free dispersion.")

            for s in np.arange(len(unique_values)):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                    if (
                        np.sum(tmp[i, idx_nonzero])
                        + np.sum(tmp[i + n_states, idx_nonzero])
                        >= 0.1
                    ):
                        model = Weighted_BetaBinom(
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
            logger.info("Updating (phasing) BAF emission parameters with shared dispersion.")

            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            for s in np.arange(len(unique_values)):
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
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
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
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            model = Weighted_BetaBinom(y, features, weights=weights, exposure=exposure)
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


def update_emission_params_bb_nophasing_uniqvalues(
    unique_values,
    mapping_matrices,
    log_gamma,
    taus,
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

    log_gamma : array, (n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.
    """
    n_spots = len(unique_values)
    n_states = log_gamma.shape[0]
    gamma = np.exp(log_gamma)
    # initialization
    new_p_binom = (
        copy.copy(start_p_binom)
        if not start_p_binom is None
        else np.ones((n_states, n_spots)) * 0.5
    )
    new_taus = copy.copy(taus)

    if fix_BB_dispersion:
        logger.info("Updating (no phasing) BB emission parameters with fixed dispersion.")

        for s in np.arange(len(unique_values)):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                if np.sum(tmp[i, idx_nonzero]) >= 0.1:
                    model = Weighted_BetaBinom_fixdispersion(
                        unique_values[s][idx_nonzero, 0],
                        np.ones(len(idx_nonzero)).reshape(-1, 1),
                        taus[i, s],
                        weights=tmp[i, idx_nonzero],
                        exposure=unique_values[s][idx_nonzero, 1],
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
            logger.info("Updating (no phasing) BB emission parameters with free dispersion.")

            for s in np.arange(len(unique_values)):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                    if np.sum(tmp[i, idx_nonzero]) >= 0.1:
                        model = Weighted_BetaBinom(
                            unique_values[s][idx_nonzero, 0],
                            np.ones(len(idx_nonzero)).reshape(-1, 1),
                            weights=tmp[i, idx_nonzero],
                            exposure=unique_values[s][idx_nonzero, 1],
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
            logger.info("Updating (no phasing) BB emission parameters with shared dispersion.")

            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            for s in np.arange(len(unique_values)):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(unique_values[s][idx_nonzero, 1], n_states)
                this_y = np.tile(unique_values[s][idx_nonzero, 0], n_states)
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
                this_weights = np.concatenate(
                    [tmp[i, idx_nonzero] for i in range(n_states)]
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
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            model = Weighted_BetaBinom(y, features, weights=weights, exposure=exposure)
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

    new_p_binom = (
        copy.copy(start_p_binom)
        if not start_p_binom is None
        else np.ones((n_states, n_spots)) * 0.5
    )
    new_taus = copy.copy(taus)

    if fix_BB_dispersion:
        logger.info("Updating (phasing, tumor mix) BAF emission parameters with fixed dispersion.")

        for s in np.arange(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
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
            logger.info("Updating (phasing, tumor mix) BAF emission parameters with free dispersion.")

            for s in np.arange(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
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
            logger.info("Updating (phasing, tumor mix) BAF emission parameters with shared dispersion.")

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
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
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


def update_emission_params_bb_nophasing_uniqvalues_mix(
    unique_values,
    mapping_matrices,
    log_gamma,
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

    log_gamma : array, (n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.
    """
    n_spots = len(unique_values)
    n_states = log_gamma.shape[0]
    gamma = np.exp(log_gamma)
    # initialization
    new_p_binom = (
        copy.copy(start_p_binom)
        if not start_p_binom is None
        else np.ones((n_states, n_spots)) * 0.5
    )
    new_taus = copy.copy(taus)
    if fix_BB_dispersion:
        logger.info("Updating (no phasing, tumor mix) BAF emission parameters with fixed dispersion.")

        for s in np.arange(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                if np.sum(tmp[i, idx_nonzero]) >= 0.1:
                    this_tp = (mapping_matrices[s].T @ tumor_prop[:, s])[
                        idx_nonzero
                    ] / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ]
                    assert np.all(this_tp < 1 + 1e-4)
                    model = Weighted_BetaBinom_fixdispersion_mix(
                        unique_values[s][idx_nonzero, 0],
                        np.ones(len(idx_nonzero)).reshape(-1, 1),
                        taus[i, s],
                        weights=tmp[i, idx_nonzero],
                        exposure=unique_values[s][idx_nonzero, 1],
                        tumor_prop=this_tp,
                    )
                    # tumor_prop=tumor_prop[s] )
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
            logger.info("Updating (no phasing, tumor mix) BAF emission parameters with free dispersion.")

            for s in np.arange(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                    if np.sum(tmp[i, idx_nonzero]) >= 0.1:
                        this_tp = (mapping_matrices[s].T @ tumor_prop[:, s])[
                            idx_nonzero
                        ] / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                            idx_nonzero
                        ]
                        assert np.all(this_tp < 1 + 1e-4)
                        model = Weighted_BetaBinom_mix(
                            unique_values[s][idx_nonzero, 0],
                            np.ones(len(idx_nonzero)).reshape(-1, 1),
                            weights=tmp[i, idx_nonzero],
                            exposure=unique_values[s][idx_nonzero, 1],
                            tumor_prop=this_tp,
                        )
                        # tumor_prop=tumor_prop[s] )
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
            logger.info("Updating (no phasing, tumor mix) BAF emission parameters with shared dispersion.")

            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            tp = []
            for s in np.arange(n_spots):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(unique_values[s][idx_nonzero, 1], n_states)
                this_y = np.tile(unique_values[s][idx_nonzero, 0], n_states)
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
                this_tp = np.tile(
                    (mapping_matrices[s].T @ tumor_prop[:, s])[idx_nonzero]
                    / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ],
                    n_states,
                )
                assert np.all(this_tp < 1 + 1e-4)
                this_weights = np.concatenate(
                    [tmp[i, idx_nonzero] for i in range(n_states)]
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
                # tp.append( tumor_prop[s] * np.ones(len(idx_row_posweight)) )
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
