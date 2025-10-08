import scipy
import pytest
import numpy as np
from scipy.stats import betabinom, nbinom
from scipy.special import loggamma
from numba import njit
from functools import partial
from cnaster.hmm_sitewise import switch_betabinom
from cnaster.hmm_emission import betabinom_logpmf, betabinom_logpmf_zp, nloglikeobs_nb


def test_phased_emission_vanilla(benchmark, baf_emission_data):
    bn, Sn, alpha, beta = baf_emission_data

    vanilla = partial(betabinom.logpmf, k=bn, n=Sn, a=beta, b=alpha)
    exp = benchmark(vanilla)


def test_phased_emission(benchmark, baf_emission_data):
    bn, Sn, alpha, beta = baf_emission_data

    original = betabinom.logpmf(bn, Sn, alpha, beta)

    vanilla = partial(betabinom.logpmf, k=bn, n=Sn, a=beta, b=alpha)
    exp = vanilla()

    tester = partial(switch_betabinom, original, bn, Sn, alpha, beta)
    result = benchmark(tester)

    # TODO assert close test exp
    np.testing.assert_allclose(result, exp, rtol=1e-10, atol=1e-12)


def test_emission_model_eval(benchmark, baf_emission_data):
    bn, Sn, alpha, beta = baf_emission_data

    exp = betabinom.logpmf(k=bn, n=Sn, a=alpha, b=beta)
    zero_point = betabinom_logpmf_zp(bn, Sn)

    solver = partial(
        betabinom_logpmf, zero_point=zero_point, endog=bn, exposure=Sn, a=alpha, b=beta
    )
    result = benchmark(solver)

    np.testing.assert_allclose(result, exp, rtol=1e-10, atol=1e-12)


@njit
def ln_rising_factorial_sorted(results, ks, r):
    n = len(ks)

    if n == 0:
        return results

    current_log_product = 0.0

    for j in range(ks[0]):
        current_log_product += np.log(r + j)

    results[0] = current_log_product
    current_k = ks[0]

    for i in range(1, n):
        k = ks[i]

        while current_k < k:
            current_log_product += np.log(r + current_k)
            current_k += 1

        results[i] = current_log_product

    return results


def ln_nb_shift(result, ks, fs, r, p):
    ln_rising_factorial_sorted(result, ks, r)

    result += result + r * np.log(p) + ks * np.log(1.0 - p) - fs

    return result.sum()


def test_nb_shift(benchmark):
    ks = np.arange(1_000)
    rs, ps = 25, 0.1

    fs = scipy.special.gammaln(1. + ks)
    result = np.empty(len(ks), dtype=np.float64)

    def run_exp():
        return -scipy.stats.nbinom.logpmf(ks, rs, ps).sum()

    def run_new():
        return -ln_nb_shift(result, ks, fs, rs, ps)

    # NB 57.2us -> 
    # exp = benchmark(run_exp)

    # NB 21.8580 -> 16.5 if sorted -> 9.25us.
    new = benchmark(run_new)


# NB 58us.
def test_nloglikeobs_nb_basic(benchmark):
    np.random.seed(42)
    
    n_samples, n_features = 1_000, 3
    mean_count = 5
    endog = np.random.poisson(mean_count, size=n_samples)
    exog = np.eye(n_features)[np.random.choice(n_features, n_samples)]
    weights = np.ones(n_samples)
    exposure = 10 + endog.copy()
    params = np.array([0.1, 0.2, 0.3, 0.5])

    exp = 3405.269453793813
    result = nloglikeobs_nb(endog, exog, weights, exposure, params)

    np.testing.assert_almost_equal(result, exp, decimal=7)

    def run():
        return nloglikeobs_nb(endog, exog, weights, exposure, params)

    benchmark(run)
