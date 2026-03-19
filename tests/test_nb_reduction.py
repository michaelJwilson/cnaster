import pytest
import numpy as np
from scipy.stats import nbinom
from scipy.optimize import minimize
from scipy.special import gammaln
from collections import namedtuple
from numba import njit

nb_params = namedtuple("nb_params", ["r", "p"])


@pytest.fixture(params=[100_000])
def mock_nb_dataset(request):
    N = request.param

    # r (number of successes) and p (probability of a single success)
    r, p = 2.5, 0.4

    ki = nbinom.rvs(r, p, size=N, random_state=42)

    return ki, nb_params(r=r, p=p)


def weighted_nll(params, ki, wi):
    r, p = params
    log_probs = nbinom.logpmf(ki, r, p)
    return -np.sum(wi * log_probs)


def test_weighted_nll(mock_nb_dataset):
    ki, (r, p) = mock_nb_dataset
    wi = np.ones_like(ki, dtype=float)

    nll = weighted_nll([r, p], ki, wi)

    # NB regression test
    exp_nll = 236702.80431

    np.testing.assert_allclose(nll, exp_nll, rtol=1e-5)


def test_mle(mock_nb_dataset, benchmark):
    ki, (exp_r, exp_p) = mock_nb_dataset
    wi = np.ones_like(ki, dtype=float)

    r, p = 1.0, 0.5
    x0 = [r, p]

    result = benchmark(
        minimize,
        weighted_nll,
        x0,
        args=(ki, wi),
        method="L-BFGS-B",
        bounds=[(1e-5, None), (1e-5, 1 - 1e-5)],
    )

    assert result.success

    np.testing.assert_allclose(result.x[0], exp_r, rtol=1e-1)
    np.testing.assert_allclose(result.x[1], exp_p, rtol=1e-1)


@njit
def get_model_vectors(kmax, r):
    base = np.arange(kmax)
    A = np.log(r + base)
    return A


@njit
def get_suffix_counts(wi, ci):
    ci_max = int(np.max(ci))
    totals = np.zeros(1 + ci_max)

    for k in range(len(ci)):
        idx = ci[k]
        totals[idx] += wi[k]

    S = np.zeros(ci_max)
    running_sum = 0.0
    for j in range(ci_max - 1, -1, -1):
        running_sum += totals[j + 1]
        S[j] = running_sum

    return S


@njit
def weighted_nll_minka(params, Sk, sum_wi, sum_ki_wi):
    r, p = params
    kmax = len(Sk)

    A = get_model_vectors(kmax, r)

    # NLL = - [ sum(Sk * A) + sum_wi * r * ln(p) + sum_ki_wi * ln(1-p) ]
    log_likelihood = (
        np.sum(Sk * A) + sum_wi * r * np.log(p) + sum_ki_wi * np.log(1.0 - p)
    )
    return -log_likelihood


def test_weighted_nll_minka(mock_nb_dataset):
    ki, (r, p) = mock_nb_dataset
    wi = np.ones_like(ki, dtype=float)

    Sk = get_suffix_counts(wi, ki.astype(np.int64))
    sum_wi = np.sum(wi)
    sum_ki_wi = np.sum(ki * wi)

    zp = np.sum(wi * gammaln(ki + 1))
    nll = zp + weighted_nll_minka([r, p], Sk, sum_wi, sum_ki_wi)

    # NB regression test
    exp_nll = weighted_nll([r, p], ki, wi)

    np.testing.assert_allclose(nll, exp_nll, rtol=1e-5)


@njit
def get_model_vectors_grad(kmax, r):
    base = np.arange(kmax)
    dA_dr = 1.0 / (r + base)
    return dA_dr


@njit
def weighted_nll_minka_jac(params, Sk, sum_wi, sum_ki_wi):
    r, p = params
    kmax = len(Sk)

    dA_dr = get_model_vectors_grad(kmax, r)

    grad_r = -(np.sum(Sk * dA_dr) + sum_wi * np.log(p))
    grad_p = -(sum_wi * r / p - sum_ki_wi / (1.0 - p))

    return np.array([grad_r, grad_p])


def test_mle_minka_grad(mock_nb_dataset, benchmark):
    ki, (exp_r, exp_p) = mock_nb_dataset
    wi = np.ones_like(ki, dtype=float)

    Sk = get_suffix_counts(wi, ki.astype(np.int64))
    sum_wi = np.sum(wi)
    sum_ki_wi = np.sum(ki * wi)

    r, p = 1.0, 0.5
    x0 = np.array([r, p])

    result = benchmark(
        minimize,
        weighted_nll_minka,
        x0,
        args=(Sk, sum_wi, sum_ki_wi),
        method="L-BFGS-B",
        jac=weighted_nll_minka_jac,
        bounds=[(1e-5, None), (1e-5, 1 - 1e-5)],
    )

    assert result.success

    np.testing.assert_allclose(result.x[0], exp_r, rtol=1e-1)
    np.testing.assert_allclose(result.x[1], exp_p, rtol=1e-1)
