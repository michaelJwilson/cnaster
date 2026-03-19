import pytest
import numpy as np
from scipy.stats import betabinom
from scipy.optimize import minimize
from collections import namedtuple
from numba import njit

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


@njit
def get_model_vectors(kmax, dkmax, nmax, alpha, beta):
    tau = alpha + beta
    base = np.arange(nmax)

    # NB len(A) = kmax; len(B) = dkmax; len(N) = nmax; i.e. 0 to kmax-1, etc.
    A = np.log(alpha + base[:kmax])
    B = np.log(beta + base[:dkmax])
    N = np.log(tau + base[:nmax])

    return A, B, N


def test_get_model_vectors(mock_bb_dataset):
    (ki, ni), (alpha, beta) = mock_bb_dataset

    kmax = int(np.max(ki))
    dkmax = int(np.max(ni - ki))
    nmax = int(np.max(ni))

    A, B, N = get_model_vectors(kmax, dkmax, nmax, alpha, beta)

    assert len(A) == kmax
    assert len(B) == dkmax
    assert len(N) == nmax


@njit
def get_window_matrix(wi, ci):
    # NB window matrix Wij = sum_i wi * 1(ci == i) for j < i; otherwise 0.
    #    W.shape = (1 + ci_max, ci_max), i.e. i = 0 ... ci_max, j = 0 ... ci_max-1;
    ci_max = int(np.max(ci))

    totals = np.zeros(1 + ci_max)
    for k in range(len(ci)):
        idx = ci[k]
        totals[idx] += wi[k]

    W = np.zeros((1 + ci_max, ci_max))
    for i in range(1 + ci_max):
        for j in range(i):
            W[i, j] = totals[i]

    return W


def test_get_window_matrix(mock_bb_dataset):
    (ki, _), _ = mock_bb_dataset
    wi = np.ones_like(ki, dtype=float)

    kmax = int(np.max(ki))
    W = get_window_matrix(wi, ki.astype(np.int64))

    assert W.shape == (1 + kmax, kmax)


def get_window_matrices(wi, ki, ni):
    Aw = get_window_matrix(wi, ki.astype(np.int64))
    Bw = get_window_matrix(wi, (ni - ki).astype(np.int64))
    Nw = get_window_matrix(wi, ni.astype(np.int64))

    return Aw, Bw, Nw


@njit
def weighted_nll_fast(params, Aw, Bw, Nw):
    alpha, beta = params

    amax = Aw.shape[1]
    bmax = Bw.shape[1]
    nmax = Nw.shape[1]

    A, B, N = get_model_vectors(amax, bmax, nmax, alpha, beta)

    return -(np.sum(Aw @ A) + np.sum(Bw @ B) - np.sum(Nw @ N))


def test_weighted_nll_fast(mock_bb_dataset):
    (ki, ni), (alpha, beta) = mock_bb_dataset
    wi = np.ones_like(ki, dtype=float)

    Aw, Bw, Nw = get_window_matrices(wi, ki, ni)

    # NB does not contain data terms, e.g. ln(x+1)
    nll = weighted_nll_fast([alpha, beta], Aw, Bw, Nw)
    exp_nll = 2876976.4476084057

    np.testing.assert_allclose(nll, exp_nll, rtol=1e-5)


def test_mle_fast(mock_bb_dataset, benchmark):
    (ki, ni), (exp_alpha, exp_beta) = mock_bb_dataset
    wi = np.ones_like(ki, dtype=float)

    Aw, Bw, Nw = get_window_matrices(wi, ki, ni)

    alpha = beta = 1.0
    x0 = np.array([alpha, beta])

    result = benchmark(
        minimize,
        weighted_nll_fast,
        x0,
        args=(Aw, Bw, Nw),
        method="L-BFGS-B",
        bounds=[(1e-5, None), (1e-5, None)],
    )

    # print(result)

    assert result.success

    np.testing.assert_allclose(result.x[0], exp_alpha, rtol=1e-1)
    np.testing.assert_allclose(result.x[1], exp_beta, rtol=1e-1)
