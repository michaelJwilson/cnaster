import pytest
import numpy as np
from scipy.stats import betabinom
from scipy.special import loggamma
from functools import partial
from cnaster.hmm_initialize import get_eff_element

def test_get_eff_element():
    exp = 17.00
    res = get_eff_element(0.9, 10)

    np.testing.assert_allclose(res, exp, rtol=1, atol=1e-2)
