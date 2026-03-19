import pytest
import numpy as np
from scipy.stats import betabinom
from scipy.optimize import minimize
from collections import namedtuple

beta_binomial_params = namedtuple("beta_binomial_params", ["alpha", "beta"])


@pytest.fixture(params=[100_000])
def mock_bb_dataset(request):
    N = request.param

    alpha, beta = 2.5, 1.5

    ni = np.random.RandomState(42).randint(1, 100, size=N)
    ki = betabinom.rvs(ni, alpha, beta, random_state=42)

    return (ki, ni), beta_binomial_params(alpha=alpha, beta=beta)


def weighted_nll(params, ki, ni, wi):
    alpha, beta = params
    log_probs = betabinom.logpmf(ki, ni, alpha, beta)
    return -np.sum(wi * log_probs)


def test_weighted_nll(mock_bb_dataset):
    (ki, ni), (alpha, beta) = mock_bb_dataset
    wi = np.ones_like(ki, dtype=float)

    nll = weighted_nll([alpha, beta], ki, ni, wi)
    exp_nll = 350941.55724

    np.testing.assert_allclose(nll, exp_nll, rtol=1e-5)


def test_mle(mock_bb_dataset, benchmark):
    (ki, ni), (exp_alpha, exp_beta) = mock_bb_dataset
    wi = np.ones_like(ki, dtype=float)

    alpha = beta = 1.0
    x0 = [alpha, beta]

    result = benchmark(
        minimize,
        weighted_nll,
        x0,
        args=(ki, ni, wi),
        method="L-BFGS-B",
        bounds=[(1e-5, None), (1e-5, None)],
    )

    assert result.success

    np.testing.assert_allclose(result.x[0], exp_alpha, rtol=1e-1)
    np.testing.assert_allclose(result.x[1], exp_beta, rtol=1e-1)


def get_model_vectors(kmax, dkmax, nmax, alpha, beta):
    tau = alpha + beta
    base = np.arange(nmax)

    A = np.log(alpha + base[kmax - 1])
    B = np.log(beta + base[dkmax - 1])
    N = np.log(tau + base[nmax - 1])

    return A, B, N


def get_window_matrix(wi, ci, ci_max):
    # NB each row is sum_i wi for ci = i;
    W = np.zeros((ci_max, ci_max))
    for i in range(ci_max):
        mask = ci == i
        result = np.sum(wi[mask])

        for j in range(i):
            W[i, j] = result

    return W


def weighted_nll_fast(params, Aw, Bw, Nw):
    alpha, beta = params

    kmax = Aw.shape[0]
    dkmax = Bw.shape[0]
    nmax = Nw.shape[0]

    A, B, N = get_model_vectors(kmax, dkmax, nmax, alpha, beta)

    return -(Aw @ A + Bw @ B - Nw @ N)


def test_mle_fast(mock_bb_dataset, benchmark):
    (ki, ni), (exp_alpha, exp_beta) = mock_bb_dataset
    wi = np.ones_like(ki, dtype=float)

    kmax = np.max(ki)
    dkmax = np.max(ni - ki)
    nmax = np.max(ni)

    Aw = get_window_matrix(wi, ki, kmax)
    Bw = get_window_matrix(wi, ni - ki, dkmax)
    Nw = get_window_matrix(wi, ni, nmax)

    alpha = beta = 1.0
    x0 = [alpha, beta]

    """
    result = benchmark(
        minimize,
        weighted_nll_fast,
        x0,
        args=(Aw, Bw, Nw),
        method="L-BFGS-B",
        bounds=[(1e-5, None), (1e-5, None)],
    )
    """

    result = minimize(
        weighted_nll_fast,
        x0,
        args=(Aw, Bw, Nw),
        method="L-BFGS-B",
        bounds=[(1e-5, None), (1e-5, None)],
    )

    assert result.success

    np.testing.assert_allclose(result.x[0], exp_alpha, rtol=1e-1)
    np.testing.assert_allclose(result.x[1], exp_beta, rtol=1e-1)
