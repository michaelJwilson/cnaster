import pytest
import numpy as np
from scipy.stats import betabinom
from scipy.special import loggamma
from functools import partial
from cnaster.hmm_sitewise import switch_betabinom
from cnaster.hmm_emission import betabinom_logpmf, betabinom_logpmf_zp

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
