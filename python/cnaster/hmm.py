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
    # prepare gmm input of RDR and BAF separately
    X_gmm_rdr = None
    X_gmm_baf = None
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
        # indexes = np.where(X_gmm_baf[:,0] > 0.5)[0]
        # X_gmm_baf[indexes,:] = 1 - X_gmm_baf[indexes,:]
        X_gmm = np.hstack([X_gmm_rdr, X_gmm_baf])
    elif "m" in params:
        X_gmm = X_gmm_rdr
    elif "p" in params:
        # indexes = np.where(X_gmm_baf[:,0] > 0.5)[0]
        # X_gmm_baf[indexes,:] = 1 - X_gmm_baf[indexes,:]
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
            gmm_p_binom = np.where(gmm_p_binom > 0.5, 1 - gmm_p_binom, gmm_p_binom)
    return gmm_log_mu, gmm_p_binom


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

    # initial log_gamma
    log_gamma = np.zeros((n_states, n_obs))

    # compute log_gamma
    log_gamma = log_alpha + log_beta

    if np.any(np.sum(log_gamma, axis=0) == 0):
        raise Exception("Sum of posterior probability is zero for some observations!")

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

    # initialize log_xi
    log_xi = np.zeros((2 * n_states, 2 * n_states, n_obs - 1))

    # compute log_xi
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
    # normalize
    for t in np.arange(n_obs - 1):
        log_xi[:, :, t] -= mylogsumexp(log_xi[:, :, t])

    return log_xi


def construct_unique_matrix(obs_count, total_count):
    """
    Attributes
    ----------
    allele_count : array, shape (n_observations, n_spots)
        Observed A allele counts per SNP per spot.

    total_bb_RD : array, shape (n_observations, n_spots)
        Total SNP-covering reads per SNP per spot.
    """
    n_obs = obs_count.shape[0]
    n_spots = obs_count.shape[1]
    unique_values = []
    mapping_matrices = []
    for s in range(n_spots):
        if total_count.dtype == int:
            pairs = np.unique(np.vstack([obs_count[:, s], total_count[:, s]]).T, axis=0)
        else:
            pairs = np.unique(
                np.vstack([obs_count[:, s], total_count[:, s]]).T.round(decimals=4),
                axis=0,
            )
        unique_values.append(pairs)
        pair_index = {(pairs[i, 0], pairs[i, 1]): i for i in range(pairs.shape[0])}
        # construct mapping matrix
        mat_row = np.arange(n_obs)
        mat_col = np.zeros(n_obs, dtype=int)
        for i in range(n_obs):
            if total_count.dtype == int:
                tmpidx = pair_index[(obs_count[i, s], total_count[i, s])]
            else:
                tmpidx = pair_index[
                    (obs_count[i, s], total_count[i, s].round(decimals=4))
                ]
            mat_col[i] = tmpidx
        mapping_matrices.append(
            scipy.sparse.csr_matrix((np.ones(len(mat_row)), (mat_row, mat_col)))
        )
    return unique_values, mapping_matrices


class hmm_sitewise(object):
    def __init__(self, params="stmp", t=1 - 1e-4):
        self.params = params
        self.t = t

    @staticmethod
    def compute_emission_probability_nb_betabinom(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
    ):
        """
        Attributes
        ----------
        X : array, shape (n_observations, n_components, n_spots)
            Observed expression UMI count and allele frequency UMI count.

        base_nb_mean : array, shape (n_observations, n_spots)
            Mean expression under diploid state.

        log_mu : array, shape (n_states, n_spots)
            Log of read depth change due to CNV. Mean of NB distributions in HMM per state per spot.

        alphas : array, shape (n_states, n_spots)
            Over-dispersion of NB distributions in HMM per state per spot.

        total_bb_RD : array, shape (n_observations, n_spots)
            SNP-covering reads for both REF and ALT across genes along genome.

        p_binom : array, shape (n_states, n_spots)
            BAF due to CNV. Mean of Beta Binomial distribution in HMM per state per spot.

        taus : array, shape (n_states, n_spots)
            Over-dispersion of Beta Binomial distribution in HMM per state per spot.

        Returns
        ----------
        log_emission : array, shape (2*n_states, n_obs, n_spots)
            Log emission probability for each gene each spot (or sample) under each state. There is a common bag of states across all spots.
        """
        n_obs = X.shape[0]
        n_comp = X.shape[1]
        n_spots = X.shape[2]
        n_states = log_mu.shape[0]

        log_emission_rdr = np.zeros((2 * n_states, n_obs, n_spots))
        log_emission_baf = np.zeros((2 * n_states, n_obs, n_spots))

        for i in np.arange(n_states):
            for s in np.arange(n_spots):
                idx_nonzero_rdr = np.where(base_nb_mean[:, s] > 0)[0]

                if len(idx_nonzero_rdr) > 0:
                    nb_mean = base_nb_mean[idx_nonzero_rdr, s] * np.exp(log_mu[i, s])
                    nb_std = np.sqrt(nb_mean + alphas[i, s] * nb_mean**2)

                    n, p = convert_params(nb_mean, nb_std)

                    log_emission_rdr[i, idx_nonzero_rdr, s] = scipy.stats.nbinom.logpmf(
                        X[idx_nonzero_rdr, 0, s], n, p
                    )

                    log_emission_rdr[
                        i + n_states, idx_nonzero_rdr, s
                    ] = log_emission_rdr[i, idx_nonzero_rdr, s]

                idx_nonzero_baf = np.where(total_bb_RD[:, s] > 0)[0]

                if len(idx_nonzero_baf) > 0:
                    log_emission_baf[
                        i, idx_nonzero_baf, s
                    ] = scipy.stats.betabinom.logpmf(
                        X[idx_nonzero_baf, 1, s],
                        total_bb_RD[idx_nonzero_baf, s],
                        p_binom[i, s] * taus[i, s],
                        (1 - p_binom[i, s]) * taus[i, s],
                    )

                    log_emission_baf[
                        i + n_states, idx_nonzero_baf, s
                    ] = scipy.stats.betabinom.logpmf(
                        X[idx_nonzero_baf, 1, s],
                        total_bb_RD[idx_nonzero_baf, s],
                        (1 - p_binom[i, s]) * taus[i, s],
                        p_binom[i, s] * taus[i, s],
                    )

        return log_emission_rdr, log_emission_baf

    @staticmethod
    def compute_emission_probability_nb_betabinom_mix(
        X,
        base_nb_mean,
        log_mu,
        alphas,
        total_bb_RD,
        p_binom,
        taus,
        tumor_prop,
        **kwargs,
    ):
        """
        Attributes
        ----------
        X : array, shape (n_observations, n_components, n_spots)
            Observed expression UMI count and allele frequency UMI count.

        base_nb_mean : array, shape (n_observations, n_spots)
            Mean expression under diploid state.

        log_mu : array, shape (n_states, n_spots)
            Log of read depth change due to CNV. Mean of NB distributions in HMM per state per spot.

        alphas : array, shape (n_states, n_spots)
            Over-dispersion of NB distributions in HMM per state per spot.

        total_bb_RD : array, shape (n_observations, n_spots)
            SNP-covering reads for both REF and ALT across genes along genome.

        p_binom : array, shape (n_states, n_spots)
            BAF due to CNV. Mean of Beta Binomial distribution in HMM per state per spot.

        taus : array, shape (n_states, n_spots)
            Over-dispersion of Beta Binomial distribution in HMM per state per spot.

        Returns
        ----------
        log_emission : array, shape (2*n_states, n_obs, n_spots)
            Log emission probability for each gene each spot (or sample) under each state. There is a common bag of states across all spots.
        """
        n_obs = X.shape[0]
        n_comp = X.shape[1]
        n_spots = X.shape[2]
        n_states = log_mu.shape[0]

        # initialize log_emission
        log_emission_rdr = np.zeros((2 * n_states, n_obs, n_spots))
        log_emission_baf = np.zeros((2 * n_states, n_obs, n_spots))

        for i in np.arange(n_states):
            for s in np.arange(n_spots):
                # expression from NB distribution
                idx_nonzero_rdr = np.where(base_nb_mean[:, s] > 0)[0]

                if len(idx_nonzero_rdr) > 0:
                    nb_mean = base_nb_mean[idx_nonzero_rdr, s] * (
                        tumor_prop[idx_nonzero_rdr, s] * np.exp(log_mu[i, s])
                        + 1
                        - tumor_prop[idx_nonzero_rdr, s]
                    )
                    nb_std = np.sqrt(nb_mean + alphas[i, s] * nb_mean**2)
                    n, p = convert_params(nb_mean, nb_std)
                    log_emission_rdr[i, idx_nonzero_rdr, s] = scipy.stats.nbinom.logpmf(
                        X[idx_nonzero_rdr, 0, s], n, p
                    )
                    log_emission_rdr[
                        i + n_states, idx_nonzero_rdr, s
                    ] = log_emission_rdr[i, idx_nonzero_rdr, s]

                # AF from BetaBinom distribution
                idx_nonzero_baf = np.where(total_bb_RD[:, s] > 0)[0]

                if len(idx_nonzero_baf) > 0:
                    mix_p_A = p_binom[i, s] * tumor_prop[idx_nonzero_baf, s] + 0.5 * (
                        1 - tumor_prop[idx_nonzero_baf, s]
                    )
                    mix_p_B = (1 - p_binom[i, s]) * tumor_prop[
                        idx_nonzero_baf, s
                    ] + 0.5 * (1 - tumor_prop[idx_nonzero_baf, s])
                    log_emission_baf[
                        i, idx_nonzero_baf, s
                    ] += scipy.stats.betabinom.logpmf(
                        X[idx_nonzero_baf, 1, s],
                        total_bb_RD[idx_nonzero_baf, s],
                        mix_p_A * taus[i, s],
                        mix_p_B * taus[i, s],
                    )
                    log_emission_baf[
                        i + n_states, idx_nonzero_baf, s
                    ] += scipy.stats.betabinom.logpmf(
                        X[idx_nonzero_baf, 1, s],
                        total_bb_RD[idx_nonzero_baf, s],
                        mix_p_B * taus[i, s],
                        mix_p_A * taus[i, s],
                    )
        return log_emission_rdr, log_emission_baf

    @staticmethod
    @njit
    def forward_lattice(
        lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
    ):
        """
        Note that n_states is the CNV states, and there are 2 * n_states of paired states for (CNV, phasing) pairs.
        Input
            lengths: sum of lengths = n_observations.
            log_transmat: n_states * n_states. Transition probability after log transformation.
            log_startprob: n_states. Start probability after log transformation.
            log_emission: 2*n_states * n_observations * n_spots. Log probability.
            log_sitewise_transmat: n_observations, the log transition probability of phase switch.
        Output
            log_alpha: size 2n_states * n_observations. log alpha[j, t] = log P(o_1, ... o_t, q_t = j | lambda).
        """
        n_obs = log_emission.shape[1]
        n_states = int(np.ceil(log_emission.shape[0] / 2))

        assert (
            np.sum(lengths) == n_obs
        ), "Sum of lengths must be equal to the first dimension of X!"

        assert (
            len(log_startprob) == n_states
        ), "Length of startprob_ must be equal to the first dimension of log_transmat!"

        log_sitewise_self_transmat = np.log(1 - np.exp(log_sitewise_transmat))

        # initialize log_alpha
        log_alpha = np.zeros((log_emission.shape[0], n_obs))
        buf = np.zeros(log_emission.shape[0])
        cumlen = 0

        for le in lengths:
            # start prob
            combined_log_startprob = np.log(0.5) + np.append(
                log_startprob, log_startprob
            )
            # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities.
            # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
            log_alpha[:, cumlen] = combined_log_startprob + np_sum_ax_squeeze(
                log_emission[:, cumlen, :], axis=1
            )
            for t in np.arange(1, le):
                phases_switch_mat = np.array(
                    [
                        [
                            log_sitewise_self_transmat[cumlen + t - 1],
                            log_sitewise_transmat[cumlen + t - 1],
                        ],
                        [
                            log_sitewise_transmat[cumlen + t - 1],
                            log_sitewise_self_transmat[cumlen + t - 1],
                        ],
                    ]
                )
                combined_transmat = np.kron(
                    np.exp(phases_switch_mat), np.exp(log_transmat)
                )
                combined_transmat = np.log(combined_transmat)
                for j in np.arange(log_emission.shape[0]):
                    for i in np.arange(log_emission.shape[0]):
                        buf[i] = (
                            log_alpha[i, (cumlen + t - 1)] + combined_transmat[i, j]
                        )
                    log_alpha[j, (cumlen + t)] = mylogsumexp(buf) + np.sum(
                        log_emission[j, (cumlen + t), :]
                    )
            cumlen += le
        return log_alpha

    @staticmethod
    @njit
    def backward_lattice(
        lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
    ):
        """
        Note that n_states is the CNV states, and there are 2 * n_states of paired states for (CNV, phasing) pairs.
        Input
            X: size n_observations * n_components * n_spots.
            lengths: sum of lengths = n_observations.
            log_transmat: n_states * n_states. Transition probability after log transformation.
            log_startprob: n_states. Start probability after log transformation.
            log_emission: 2*n_states * n_observations * n_spots. Log probability.
            log_sitewise_transmat: n_observations, the log transition probability of phase switch.
        Output
            log_beta: size 2*n_states * n_observations. log beta[i, t] = log P(o_{t+1}, ..., o_T | q_t = i, lambda).
        """
        n_obs = log_emission.shape[1]
        n_states = int(np.ceil(log_emission.shape[0] / 2))
        assert (
            np.sum(lengths) == n_obs
        ), "Sum of lengths must be equal to the first dimension of X!"
        assert (
            len(log_startprob) == n_states
        ), "Length of startprob_ must be equal to the first dimension of log_transmat!"
        log_sitewise_self_transmat = np.log(1 - np.exp(log_sitewise_transmat))
        # initialize log_beta
        log_beta = np.zeros((log_emission.shape[0], n_obs))
        buf = np.zeros(log_emission.shape[0])
        cumlen = 0
        for le in lengths:
            # start prob
            # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities.
            # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
            log_beta[:, (cumlen + le - 1)] = 0
            for t in np.arange(le - 2, -1, -1):
                phases_switch_mat = np.array(
                    [
                        [
                            log_sitewise_self_transmat[cumlen + t],
                            log_sitewise_transmat[cumlen + t],
                        ],
                        [
                            log_sitewise_transmat[cumlen + t],
                            log_sitewise_self_transmat[cumlen + t],
                        ],
                    ]
                )
                combined_transmat = np.kron(
                    np.exp(phases_switch_mat), np.exp(log_transmat)
                )
                combined_transmat = np.log(combined_transmat)
                for i in np.arange(log_emission.shape[0]):
                    for j in np.arange(log_emission.shape[0]):
                        buf[j] = (
                            log_beta[j, (cumlen + t + 1)]
                            + combined_transmat[i, j]
                            + np.sum(log_emission[j, (cumlen + t + 1), :])
                        )
                    log_beta[i, (cumlen + t)] = mylogsumexp(buf)
            cumlen += le
        return log_beta

    def run_baum_welch_nb_bb(
        self,
        X,
        lengths,
        n_states,
        base_nb_mean,
        total_bb_RD,
        log_sitewise_transmat,
        tumor_prop=None,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        is_diag=False,
        init_log_mu=None,
        init_p_binom=None,
        init_alphas=None,
        init_taus=None,
        max_iter=100,
        tol=1e-4,
    ):
        """
        Input
            X: size n_observations * n_components * n_spots.
            lengths: sum of lengths = n_observations.
            base_nb_mean: size of n_observations * n_spots.
            In NB-BetaBinom model, n_components = 2
        Intermediate
            log_mu: size of n_states. Log of mean/exposure/base_prob of each HMM state.
            alpha: size of n_states. Dispersioon parameter of each HMM state.
        """
        n_obs = X.shape[0]
        n_comp = X.shape[1]
        n_spots = X.shape[2]
        assert n_comp == 2
        # initialize NB logmean shift and BetaBinom prob
        log_mu = (
            np.vstack([np.linspace(-0.1, 0.1, n_states) for r in range(n_spots)]).T
            if init_log_mu is None
            else init_log_mu
        )
        p_binom = (
            np.vstack([np.linspace(0.05, 0.45, n_states) for r in range(n_spots)]).T
            if init_p_binom is None
            else init_p_binom
        )
        # initialize (inverse of) dispersion param in NB and BetaBinom
        alphas = (
            0.1 * np.ones((n_states, n_spots)) if init_alphas is None else init_alphas
        )
        taus = 30 * np.ones((n_states, n_spots)) if init_taus is None else init_taus

        # initialize start probability and emission probability
        log_startprob = np.log(np.ones(n_states) / n_states)

        if n_states > 1:
            transmat = np.ones((n_states, n_states)) * (1 - self.t) / (n_states - 1)
            np.fill_diagonal(transmat, self.t)
            log_transmat = np.log(transmat)
        else:
            log_transmat = np.zeros((1, 1))

        # a trick to speed up BetaBinom optimization: taking only unique values of (B allele count, total SNP covering read count)
        unique_values_nb, mapping_matrices_nb = construct_unique_matrix(
            X[:, 0, :], base_nb_mean
        )

        unique_values_bb, mapping_matrices_bb = construct_unique_matrix(
            X[:, 1, :], total_bb_RD
        )

        for r in trange(max_iter):
            if tumor_prop is None:
                (
                    log_emission_rdr,
                    log_emission_baf,
                ) = hmm_sitewise.compute_emission_probability_nb_betabinom(
                    X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
                )
                log_emission = log_emission_rdr + log_emission_baf
            else:
                (
                    log_emission_rdr,
                    log_emission_baf,
                ) = hmm_sitewise.compute_emission_probability_nb_betabinom_mix(
                    X,
                    base_nb_mean,
                    log_mu,
                    alphas,
                    total_bb_RD,
                    p_binom,
                    taus,
                    tumor_prop,
                )
                log_emission = log_emission_rdr + log_emission_baf

            log_alpha = hmm_sitewise.forward_lattice(
                lengths,
                log_transmat,
                log_startprob,
                log_emission,
                log_sitewise_transmat,
            )

            log_beta = hmm_sitewise.backward_lattice(
                lengths,
                log_transmat,
                log_startprob,
                log_emission,
                log_sitewise_transmat,
            )

            log_gamma = compute_posterior_obs(log_alpha, log_beta)

            log_xi = compute_posterior_transition_sitewise(
                log_alpha, log_beta, log_transmat, log_emission
            )

            # M step
            if "s" in self.params:
                new_log_startprob = update_startprob_sitewise(lengths, log_gamma)
                new_log_startprob = new_log_startprob.flatten()
            else:
                new_log_startprob = log_startprob
            if "t" in self.params:
                new_log_transmat = update_transition_sitewise(log_xi, is_diag=is_diag)
            else:
                new_log_transmat = log_transmat
            if "m" in self.params:
                # new_log_mu, new_alphas = update_emission_params_nb_sitewise(X[:,0,:], log_gamma, base_nb_mean, alphas, start_log_mu=log_mu, \
                #     fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion)
                if tumor_prop is None:
                    (
                        new_log_mu,
                        new_alphas,
                    ) = update_emission_params_nb_sitewise_uniqvalues(
                        unique_values_nb,
                        mapping_matrices_nb,
                        log_gamma,
                        base_nb_mean,
                        alphas,
                        start_log_mu=log_mu,
                        fix_NB_dispersion=fix_NB_dispersion,
                        shared_NB_dispersion=shared_NB_dispersion,
                    )
                else:
                    (
                        new_log_mu,
                        new_alphas,
                    ) = update_emission_params_nb_sitewise_uniqvalues_mix(
                        unique_values_nb,
                        mapping_matrices_nb,
                        log_gamma,
                        base_nb_mean,
                        alphas,
                        tumor_prop,
                        start_log_mu=log_mu,
                        fix_NB_dispersion=fix_NB_dispersion,
                        shared_NB_dispersion=shared_NB_dispersion,
                    )
            else:
                new_log_mu = log_mu
                new_alphas = alphas
            if "p" in self.params:
                if tumor_prop is None:
                    (
                        new_p_binom,
                        new_taus,
                    ) = update_emission_params_bb_sitewise_uniqvalues(
                        unique_values_bb,
                        mapping_matrices_bb,
                        log_gamma,
                        total_bb_RD,
                        taus,
                        start_p_binom=p_binom,
                        fix_BB_dispersion=fix_BB_dispersion,
                        shared_BB_dispersion=shared_BB_dispersion,
                    )
                else:
                    (
                        new_p_binom,
                        new_taus,
                    ) = update_emission_params_bb_sitewise_uniqvalues_mix(
                        unique_values_bb,
                        mapping_matrices_bb,
                        log_gamma,
                        total_bb_RD,
                        taus,
                        tumor_prop,
                        start_p_binom=p_binom,
                        fix_BB_dispersion=fix_BB_dispersion,
                        shared_BB_dispersion=shared_BB_dispersion,
                    )
            else:
                new_p_binom = p_binom
                new_taus = taus
            # check convergence
            print(
                np.mean(np.abs(np.exp(new_log_startprob) - np.exp(log_startprob))),
                np.mean(np.abs(np.exp(new_log_transmat) - np.exp(log_transmat))),
                np.mean(np.abs(new_log_mu - log_mu)),
                np.mean(np.abs(new_p_binom - p_binom)),
            )
            print(np.hstack([new_log_mu, new_p_binom]))
            if (
                np.mean(np.abs(np.exp(new_log_transmat) - np.exp(log_transmat))) < tol
                and np.mean(np.abs(new_log_mu - log_mu)) < tol
                and np.mean(np.abs(new_p_binom - p_binom)) < tol
            ):
                break
            log_startprob = new_log_startprob
            log_transmat = new_log_transmat
            log_mu = new_log_mu
            alphas = new_alphas
            p_binom = new_p_binom
            taus = new_taus
        return (
            new_log_mu,
            new_alphas,
            new_p_binom,
            new_taus,
            new_log_startprob,
            new_log_transmat,
            log_gamma,
        )


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

    print(f"init_log_mu = {init_log_mu}")
    print(f"init_p_binom = {init_p_binom}")

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
        # log_emission_rdr, log_emission_baf = hmmclass.compute_emission_probability_nb_betabinom_mix(X, base_nb_mean, new_log_mu, new_alphas, total_bb_RD, new_p_binom, new_taus, tumor_prop)
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

    # save results
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
