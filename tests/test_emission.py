import pytest
import numpy as np
from scipy.stats import betabinom
from scipy.special import loggamma
from functools import partial
from cnaster.hmm_sitewise import switch_betabinom


def betabinom_zp_eval(endog, exposure, a, b):
    return loggamma(exposure + 1) - loggamma(endog + 1) - loggamma(exposure - endog + 1)
    
def betabinom_eval(endog, exposure, a, b, zero_point):
    # NB model dependent BB factors;                                                                                                                                                        
    #    exp: zero_point = loggamma(Sn + 1) - loggamma(bn + 1) - loggamma(Sn - bn + 1)                                                                                                      
    #         i.e. 3/9 of gamma computations.                                                                                                                                               
    result = (
        zero_point
        + loggamma(endog + a)
        + loggamma(exposure - endog + b)
        + loggamma(a + b)
        - loggamma(exposure + a + b)
        - loggamma(a)
        - loggamma(b)
    )
    return result


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
    zero_point = betabinom_zp_eval(bn, Sn, alpha, beta)
    
    solver = partial(
        betabinom_eval, zero_point=zero_point, endog=bn, exposure=Sn, a=alpha, b=beta
    )
    result = benchmark(solver)

    np.testing.assert_allclose(result, exp, rtol=1e-10, atol=1e-12)
