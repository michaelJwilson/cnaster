import sys
import numpy as np
import matplotlib.pyplot as plt
import corner
from numba import njit
from math import lgamma
from cnaster.hmm_utils import convert_params_disp


@njit(cache=True)
def numba_nloglikeobs_nb(
    endog,
    exog,
    weights,
    exposure,
    params,
    tumor_prop,
    zero_point,
):
    coeffs = np.exp(params[:-1])
    nb_mean = (exog @ coeffs) * exposure

    n_val, p = convert_params_disp(nb_mean, params[-1])

    result = np.empty_like(endog, dtype=np.float64)

    for i in range(len(endog)):
        k = endog[i]
        pi = p[i]

        # logpmf = lgamma(k + n) - lgamma(n) - lgamma(k + 1) + n * log(p) + k * log(1 - p)
        log_pmf = (
            lgamma(k + n_val)
            - lgamma(n_val)
            - lgamma(k + 1.0)
            + n_val * np.log(pi)
            + k * np.log(1.0 - pi)
        )

        if np.isnan(log_pmf):
            result[i] = np.inf
        else:
            result[i] = -log_pmf

    return result.dot(weights)


@njit(nogil=True, cache=True, fastmath=False, error_model="numpy")
def betabinom_logpmf(endog, exposure, a, b, zero_point):
    result_array = np.empty_like(endog, dtype=np.float64)

    for i in range(len(endog)):
        result_array[i] = (
            zero_point[i]
            + lgamma(endog[i] + a[i])
            + lgamma(exposure[i] - endog[i] + b[i])
            + lgamma(a[i] + b[i])
            - lgamma(exposure[i] + a[i] + b[i])
            - lgamma(a[i])
            - lgamma(b[i])
        )
        if np.isnan(result_array[i]):
            result_array[i] = np.inf

    return result_array


@njit(nogil=True, cache=True, fastmath=False, error_model="numpy")
def compute_bb_ab(exog, params, tumor_prop=None):
    p = np.dot(exog, params[:-1])
    tau = params[-1]

    if tumor_prop is None:
        a = p * tau
        b = (1.0 - p) * tau
    else:
        a = (p * tumor_prop + 0.5 * (1.0 - tumor_prop)) * tau
        b = ((1.0 - p) * tumor_prop + 0.5 * (1.0 - tumor_prop)) * tau

    return a, b


@njit(cache=True)
def numba_nloglikeobs_bb(
    endog,
    exog,
    weights,
    exposure,
    params,
    tumor_prop,
    zero_point,
):
    a, b = compute_bb_ab(exog, params, tumor_prop)
    result = -betabinom_logpmf(endog, exposure, a, b, zero_point)

    return result.dot(weights)


@njit(cache=True)
def run_mcmc_numba(
    nloglikeob,
    start_params,
    n_samples,
    burn_in,
    endog,
    exog,
    weights,
    exposure,
    tumor_prop,
    zero_point,
    bounds,
    step_scales,
):
    current_params = start_params.copy()
    n_params = len(current_params)

    current_nloglikeobs = nloglikeob(
        endog, exog, weights, exposure, current_params, tumor_prop, zero_point
    )

    samples = np.empty((n_samples, n_params))
    accepted = 0

    target_acceptance = 0.60
    adaptation_window = 1_000
    batch_accepted = 0

    for i in range(n_samples + burn_in):
        proposal = current_params + np.random.standard_normal(n_params) * step_scales

        valid = True
        for idx in range(n_params):
            val = proposal[idx]
            if val < bounds[idx, 0] or val > bounds[idx, 1]:
                valid = False
                break

        if not valid:
            if i >= burn_in:
                samples[i - burn_in] = current_params
            continue

        prop_nloglikeobs = nloglikeob(
            endog, exog, weights, exposure, proposal, tumor_prop, zero_point
        )

        if np.log(np.random.rand()) < (current_nloglikeobs - prop_nloglikeobs):
            current_params = proposal
            current_nloglikeobs = prop_nloglikeobs
            if i < burn_in:
                batch_accepted += 1
            if i >= burn_in:
                accepted += 1

        if i < burn_in and (i + 1) % adaptation_window == 0:
            batch_acceptance_rate = batch_accepted / adaptation_window

            if batch_acceptance_rate > target_acceptance:
                step_scales *= 1.05
            else:
                step_scales *= 0.95

            batch_accepted = 0

        if i >= burn_in:
            samples[i - burn_in] = current_params

    return samples, accepted


def plot_mcmc(samples, labels, prefix, optimum=None):
    # means = np.mean(samples, axis=0)
    # errors = np.std(samples, axis=0)

    n_params = samples.shape[1]

    # NB scale samples (and optimum) for visualization
    samples_plot = samples.copy()
    samples_plot[:, -1] /= 1.0e3

    optimum_plot = None

    """
    if optimum is not None:
        optimum_plot = optimum.copy()
        optimum_plot[-1] /= 1.0e3
    """
    fig = plt.figure(figsize=(2.5 * n_params, 2.5 * n_params))
    fig = corner.corner(
        samples_plot,
        fig=fig,
        labels=labels,
        show_titles=True,
        title_fmt=".3f",
        top_ticks=False,
        plot_datapoints=False,
        color="#A3C1AD",
        label_kwargs={"fontsize": 11},
        title_kwargs={"fontsize": 11},
    )
    plt.savefig(f"{prefix}_mcmc.pdf")
    sys.exit(0)
