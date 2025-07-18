import logging
import warnings
import time

import numpy as np
import scipy.stats
from cnaster.config import YAMLConfig
from cnaster.hmm_utils import convert_params
from statsmodels.base.model import ValueWarning
from statsmodels.base.model import GenericLikelihoodModel


logger = logging.getLogger(__name__)

# TODO
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

# TODO HACK
SOLVER = get_global_config().hmm.solver

known_solvers = ("newton", "bfgs", "lbfgs", "powell", "nm", "cg", "ncg")

assert SOLVER in known_solvers, f"Unknown solver: {SOLVER}. Supported solvers: {known_solvers}"

class Weighted_NegativeBinomial(GenericLikelihoodModel):
    """
    Negative Binomial model endog ~ NB(exposure * exp(exog @ params[:-1]), params[-1]), where exog is the design matrix, and params[-1] is 1 / overdispersion.
    This function fits the NB params when samples are weighted by weights: max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)

    Attributes
    ----------
    endog : array, (n_samples,)
        Y values.

    exog : array, (n_samples, n_features)
        Design matrix.

    weights : array, (n_samples,)
        Sample weights.

    exposure : array, (n_samples,)
        Multiplication constant outside the exponential term. In scRNA-seq or SRT data, this term is the total UMI count per cell/spot.
    """

    def __init__(self, endog, exog, weights, exposure, seed=0, **kwds):
        super().__init__(endog, exog, **kwds)

        self.weights = weights
        self.exposure = exposure
        self.seed = seed

    def nloglikeobs(self, params):
        nb_mean = np.exp(self.exog @ params[:-1]) * self.exposure
        nb_std = np.sqrt(nb_mean + params[-1] * nb_mean**2)

        n, p = convert_params(nb_mean, nb_std)

        llf = scipy.stats.nbinom.logpmf(self.endog, n, p)

        return -llf.dot(self.weights)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        using_default_params = start_params is None
        
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                start_params = np.append(0.1 * np.ones(self.nparams), 0.01)

        start_time = time.time()
        result = super().fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, method=SOLVER, **kwds
        )
        runtime = time.time() - start_time

        logger.debug(
            f"Weighted_NegativeBinomial debug - mle_retvals: {result.mle_retvals}, "
            f"mle_settings: {result.mle_settings}"
        )

        logger.info(
            f"Weighted_NegativeBinomial done: {runtime:.2f}s, "
            f"{len(start_params)} params ({'with default start' if using_default_params else 'with custom start'}), "
            f"{result.mle_retvals.get('iterations', 'N/A')} iter, "
            f"{result.mle_retvals.get('fcalls', 'N/A')} fcalls, "
            f"optimizer: {result.mle_settings.get('optimizer', 'Unknown')}, "
            f"converged: {result.mle_retvals.get('converged', 'N/A')}, "
            f"llf: {result.llf:.6e}"
        )

        return result


class Weighted_NegativeBinomial_mix(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights, exposure, tumor_prop, seed=0, **kwds):
        super().__init__(endog, exog, **kwds)

        self.weights = weights
        self.exposure = exposure
        self.seed = seed
        self.tumor_prop = tumor_prop

    def nloglikeobs(self, params):
        nb_mean = self.exposure * (
            self.tumor_prop * np.exp(self.exog @ params[:-1]) + 1 - self.tumor_prop
        )
        nb_std = np.sqrt(nb_mean + params[-1] * nb_mean**2)

        n, p = convert_params(nb_mean, nb_std)

        llf = scipy.stats.nbinom.logpmf(self.endog, n, p)

        return -llf.dot(self.weights)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        using_default_params = start_params is None
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                start_params = np.append(0.1 * np.ones(self.nparams), 0.01)

        start_time = time.time()
        result = super().fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, method=SOLVER, **kwds
        )
        runtime = time.time() - start_time

        logger.debug(
            f"Weighted_NegativeBinomial_mix debug - mle_retvals: {result.mle_retvals}, "
            f"mle_settings: {result.mle_settings}"
        )

        logger.info(
            f"Weighted_NegativeBinomial_mix done: {runtime:.2f}s, "
            f"{len(start_params)} params ({'with default start' if using_default_params else 'with custom start'}), "
            f"{result.mle_retvals.get('iterations', 'N/A')} iter, "
            f"{result.mle_retvals.get('fcalls', 'N/A')} fcalls, "
            f"optimizer: {result.mle_settings.get('optimizer', 'Unknown')}, "
            f"converged: {result.mle_retvals.get('converged', 'N/A')}, "
            f"llf: {result.llf:.6e}"
        )

        return result


class Weighted_BetaBinom(GenericLikelihoodModel):
    """
    Beta-binomial model endog ~ BetaBin(exposure, tau * p, tau * (1 - p)), where p = exog @ params[:-1] and tau = params[-1].
    This function fits the BetaBin params when samples are weighted by weights: max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)

    Attributes
    ----------
    endog : array, (n_samples,)
        Y values.

    exog : array, (n_samples, n_features)
        Design matrix.

    weights : array, (n_samples,)
        Sample weights.

    exposure : array, (n_samples,)
        Total number of trials. In BAF case, this is the total number of SNP-covering UMIs.
    """

    def __init__(self, endog, exog, weights, exposure, **kwds):
        super().__init__(endog, exog, **kwds)
        self.weights = weights
        self.exposure = exposure

    def nloglikeobs(self, params):
        a = (self.exog @ params[:-1]) * params[-1]
        b = (1 - self.exog @ params[:-1]) * params[-1]

        llf = scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b)

        return -llf.dot(self.weights)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        using_default_params = start_params is None
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                start_params = np.append(
                    0.5 / np.sum(self.exog.shape[1]) * np.ones(self.nparams), 1
                )

        start_time = time.time()
        result = super().fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, method=SOLVER, **kwds
        )
        runtime = time.time() - start_time

        logger.debug(
            f"Weighted_BetaBinom debug - mle_retvals: {result.mle_retvals}, "
            f"mle_settings: {result.mle_settings}"
        )

        logger.info(
            f"Weighted_BetaBinom done: {runtime:.2f}s, "
            f"{len(start_params)} params ({'with default start' if using_default_params else 'with custom start'}), "
            f"{result.mle_retvals.get('iterations', 'N/A')} iter, "
            f"{result.mle_retvals.get('fcalls', 'N/A')} fcalls, "
            f"optimizer: {result.mle_settings.get('optimizer', 'Unknown')}, "
            f"converged: {result.mle_retvals.get('converged', 'N/A')}, "
            f"llf: {result.llf:.6e}"
        )

        return result


class Weighted_BetaBinom_mix(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights, exposure, tumor_prop, **kwds):
        super().__init__(endog, exog, **kwds)
        self.weights = weights
        self.exposure = exposure
        self.tumor_prop = tumor_prop

    def nloglikeobs(self, params):
        a = (
            self.exog @ params[:-1] * self.tumor_prop + 0.5 * (1 - self.tumor_prop)
        ) * params[-1]
        b = (
            (1 - self.exog @ params[:-1]) * self.tumor_prop
            + 0.5 * (1 - self.tumor_prop)
        ) * params[-1]

        llf = scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b)

        return -llf.dot(self.weights)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        using_default_params = start_params is None
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                start_params = np.append(
                    0.5 / np.sum(self.exog.shape[1]) * np.ones(self.nparams), 1
                )

        start_time = time.time()
        result = super().fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, method=SOLVER, **kwds
        )
        runtime = time.time() - start_time

        logger.debug(
            f"Weighted_BetaBinom_mix debug - mle_retvals: {result.mle_retvals}, "
            f"mle_settings: {result.mle_settings}"
        )

        logger.info(
            f"Weighted_BetaBinom_mix done: {runtime:.2f}s, "
            f"{len(start_params)} params ({'with default start' if using_default_params else 'with custom start'}), "
            f"{result.mle_retvals.get('iterations', 'N/A')} iter, "
            f"{result.mle_retvals.get('fcalls', 'N/A')} fcalls, "
            f"optimizer: {result.mle_settings.get('optimizer', 'Unknown')}, "
            f"converged: {result.mle_retvals.get('converged', 'N/A')}, "
            f"llf: {result.llf:.6e}"
        )

        return result
