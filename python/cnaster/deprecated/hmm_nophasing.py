import scipy
import numpy as np
from cnaster.hmm_utils import convert_params

def compute_emission_probability_nb_betabinom(
    X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
):
    n_obs, n_comp, n_spots = X.shape
    n_states = log_mu.shape[0]

    log_emission_rdr = np.zeros((n_states, n_obs, n_spots))
    log_emission_baf = np.zeros((n_states, n_obs, n_spots))

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

            idx_nonzero_baf = np.where(total_bb_RD[:, s] > 0)[0]

            if len(idx_nonzero_baf) > 0:
                log_emission_baf[i, idx_nonzero_baf, s] = scipy.stats.betabinom.logpmf(
                    X[idx_nonzero_baf, 1, s],
                    total_bb_RD[idx_nonzero_baf, s],
                    p_binom[i, s] * taus[i, s],
                    (1.0 - p_binom[i, s]) * taus[i, s],
                )

    return log_emission_rdr, log_emission_baf
