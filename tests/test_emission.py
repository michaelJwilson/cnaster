import pytest
import numpy as np
from scipy.stats import betabinom
from scipy.special import loggamma
from functools import partial
from cnaster.hmm_sitewise import switch_betabinom

@pytest.fixture
def emission_data():
    # NB one typical phasing spot,                                                                                                                                                                                   
    #    see https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html                                                                                                                       
    X = np.random.randint(low=0, high=20, size=((4494, 2, 1)))
    total_bb_RD = 100 + X[:, 0, :]

    p_binom, tau = 0.25, 25.0
    alpha, beta = p_binom * tau, (1.0 - p_binom) * tau

    bn, Sn = X[:, 1, :], total_bb_RD

    return bn, Sn, alpha, beta 

def test_phased_emission_vanilla(benchmark, emission_data):
    bn, Sn, alpha, beta = emission_data

    vanilla = partial(betabinom.logpmf, k=bn, n=Sn, a=beta, b=alpha)
    exp = benchmark(vanilla)
    
def test_phased_emission(benchmark, emission_data):
    bn, Sn, alpha, beta = emission_data
    
    original = betabinom.logpmf(bn, Sn, alpha, beta)

    vanilla = partial(betabinom.logpmf, k=bn, n=Sn, a=beta, b=alpha)
    exp = vanilla()
    
    tester = partial(switch_betabinom, original, bn, Sn, alpha, beta)
    result = benchmark(tester)
    
    # TODO assert close test exp
    np.testing.assert_allclose(result, exp, rtol=1e-10, atol=1e-12)
