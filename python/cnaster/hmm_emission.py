import logging
import warnings

import numpy as np
import scipy.stats
from cnaster.hmm_utils import convert_params
from statsmodels.base.model import ValueWarning
from statsmodels.base.model import GenericLikelihoodModel


logger = logging.getLogger(__name__)

# TODO
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


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
        super(Weighted_NegativeBinomial, self).__init__(endog, exog, **kwds)
        
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
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                start_params = np.append(0.1 * np.ones(self.nparams), 0.01)

        logger.info(f"Starting Weighted_NegativeBinomial fit with {len(start_params)} parameters")
        result = super(Weighted_NegativeBinomial, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds
        )
        
        logger.info(f"Weighted_NegativeBinomial fit completed - Iterations: {result.mle_retvals.get('iterations', 'N/A')}, "
                   f"Converged: {result.mle_retvals.get('converged', 'N/A')}, "
                   f"Function evaluations: {result.mle_retvals.get('funcalls', 'N/A')}")
        
        return result


class Weighted_NegativeBinomial_mix(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights, exposure, tumor_prop, seed=0, **kwds):
        super(Weighted_NegativeBinomial_mix, self).__init__(endog, exog, **kwds)

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
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                start_params = np.append(0.1 * np.ones(self.nparams), 0.01)
        
        logger.info(f"Starting Weighted_NegativeBinomial_mix fit with {len(start_params)} parameters")
        result = super(Weighted_NegativeBinomial_mix, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds
        )
        
        logger.info(f"Weighted_NegativeBinomial_mix fit completed - Iterations: {result.mle_retvals.get('iterations', 'N/A')}, "
                   f"Converged: {result.mle_retvals.get('converged', 'N/A')}, "
                   f"Function evaluations: {result.mle_retvals.get('funcalls', 'N/A')}")
        
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
        super(Weighted_BetaBinom, self).__init__(endog, exog, **kwds)
        self.weights = weights
        self.exposure = exposure

    def nloglikeobs(self, params):
        a = (self.exog @ params[:-1]) * params[-1]
        b = (1 - self.exog @ params[:-1]) * params[-1]
        
        llf = scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b)
        
        return -llf.dot(self.weights)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                start_params = np.append(
                    0.5 / np.sum(self.exog.shape[1]) * np.ones(self.nparams), 1
                )
        
        logger.info(f"Starting Weighted_BetaBinom fit with {len(start_params)} parameters")
        result = super(Weighted_BetaBinom, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds
        )
        
        logger.info(f"Weighted_BetaBinom fit completed - Iterations: {result.mle_retvals.get('iterations', 'N/A')}, "
                   f"Converged: {result.mle_retvals.get('converged', 'N/A')}, "
                   f"Function evaluations: {result.mle_retvals.get('funcalls', 'N/A')}")
        
        return result


class Weighted_BetaBinom_mix(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights, exposure, tumor_prop, **kwds):
        super(Weighted_BetaBinom_mix, self).__init__(endog, exog, **kwds)
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
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                start_params = np.append(
                    0.5 / np.sum(self.exog.shape[1]) * np.ones(self.nparams), 1
                )
        
        logger.info(f"Starting Weighted_BetaBinom_mix fit with {len(start_params)} parameters")
        result = super(Weighted_BetaBinom_mix, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds
        )
        
        logger.info(f"Weighted_BetaBinom_mix fit completed - Iterations: {result.mle_retvals.get('iterations', 'N/A')}, "
                   f"Converged: {result.mle_retvals.get('converged', 'N/A')}, "
                   f"Function evaluations: {result.mle_retvals.get('funcalls', 'N/A')}")
        
        return result
