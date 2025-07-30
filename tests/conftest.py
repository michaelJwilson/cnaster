from __future__ import annotations

import numpy as np
import pytest


# NB scope defines the event for which a new instance is generated.
#    e.g. for every module, of every test function.
@pytest.fixture
def rng():
    return np.random.default_rng(314)


@pytest.fixture
def rdr_baf(rng):
    return 5 * (1.0 + rng.uniform(size=(3, 2)))


@pytest.fixture
def baf_emission_data():
    # NB one typical phasing spot,
    #    see https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
    X = np.random.randint(low=0, high=20, size=((4494, 2, 1)))
    total_bb_RD = 100 + X[:, 0, :]

    p_binom, tau = 0.25, 25.0
    alpha, beta = p_binom * tau, (1.0 - p_binom) * tau

    bn, Sn = X[:, 1, :], total_bb_RD

    return bn, Sn, alpha, beta


def pytest_addoption(parser):
    parser.addoption(
        "--run_slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_slow"):
        return

    skip_slow = pytest.mark.skip(reason="need --run_slow option to run")

    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
