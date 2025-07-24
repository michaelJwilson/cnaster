import logging
import time
import warnings

import numpy as np
import scipy.stats
from functools import partial
from cnaster.config import get_global_config
from cnaster.hmm_utils import convert_params, get_solver

# from cnaster.deprecated.hmm_emission import Weighted_NegativeBinomial, Weighted_BetaBinom
from statsmodels.base.model import GenericLikelihoodModel

logger = logging.getLogger(__name__)

# TODO
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


class Weighted_NegativeBinomial_mix(GenericLikelihoodModel):
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

    def __init__(
        self, endog, exog, weights, exposure, tumor_prop=None, seed=0, **kwargs
    ):
        super().__init__(endog, exog, **kwargs)

        # NB EM-based posterior weights.
        self.weights = weights
        self.exposure = exposure
        self.seed = seed
        self.tumor_prop = tumor_prop

    def nloglikeobs(self, params):
        if self.tumor_prop is None:
            nb_mean = self.exog @ np.exp(params[:-1]) * self.exposure
        else:
            nb_mean = self.exposure * (
                self.tumor_prop * self.exog @ np.exp(params[:-1])
                + (1.0 - self.tumor_prop)
            )

        nb_std = np.sqrt(nb_mean + params[-1] * nb_mean**2)

        n, p = convert_params(nb_mean, nb_std)
        """
        # TODO HACK Weighted_NegativeBinomial_mix achievable compression: 80.00 as not spot aggregated.
        counts = np.vstack([self.endog, n, p]).T

        if counts.dtype != int:
            counts = counts.round(decimals=4)

        pairs = np.unique(counts, axis=0)

        mean_compression = (1. - len(pairs) / len(self.endog))

        if mean_compression > 0.1:
            logger.warning(f"TODO: {self.__class__.__name__} achievable compression: {100. * mean_compression:.4f}")
            exit(0)
        """

        return -scipy.stats.nbinom.logpmf(self.endog, n, p).dot(self.weights)

    def fit(self, start_params=None, maxiter=10_000, maxfun=5_000, **kwargs):
        using_default_params = start_params is None
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                start_params = np.append(0.1 * np.ones(self.nparams), 1.0e-2)

        start_time = time.time()
        result = super().fit(
            start_params=start_params,
            maxiter=maxiter,
            maxfun=maxfun,
            method=get_solver(),
            **kwargs,
        )
        runtime = time.time() - start_time

        logger.debug(
            f"Weighted_NegativeBinomial_mix debug - mle_retvals: {result.mle_retvals}, "
            f"mle_settings: {result.mle_settings}"
        )

        logger.info(
            f"Weighted_NegativeBinomial_mix done: {runtime:.2f}s with tumor_prop={self.tumor_prop is not None}, "
            f"{len(start_params)} params ({'with default start' if using_default_params else 'with custom start'}), "
            f"{result.mle_retvals.get('iterations', 'N/A')} iter, "
            f"{result.mle_retvals.get('fcalls', 'N/A')} fcalls, "
            f"optimizer: {result.mle_settings.get('optimizer', 'Unknown')}, "
            f"converged: {result.mle_retvals.get('converged', 'N/A')}, "
            f"llf: {result.llf:.6e}"
        )

        return result


class Weighted_BetaBinom_mix(GenericLikelihoodModel):
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

    def __init__(
        self, endog, exog, weights, exposure, tumor_prop=None, compress=False, **kwargs
    ):
        super().__init__(endog, exog, **kwargs)

        # NB EM-based posterior weights.
        self.weights = weights
        self.exposure = exposure
        self.tumor_prop = tumor_prop

        if compress:
            if tumor_prop is not None:
                logger.warning(
                    f"{self.__class__} compression is not supported for tumor_prop != None."
                )
                raise RuntimeError()

            cls = np.argmax(self.exog, axis=1)
            counts = np.vstack([self.endog, self.exposure, cls]).T

            if counts.dtype != int:
                counts = counts.round(decimals=4)

            # NB see https://numpy.org/doc/stable/reference/generated/numpy.unique.html
            unique_pairs, unique_idx, unique_inv = np.unique(
                counts, return_index=True, return_inverse=True, axis=0
            )
            mean_compression = 1.0 - len(unique_pairs) / len(self.endog)

            logger.warning(
                f"TODO: {self.__class__.__name__} achievable compression: {100. * mean_compression:.4f}"
            )

            # TODO HACK
            # NB sum self.weights
            transfer = np.zeros((len(unique_pairs), len(self.endog)), dtype=int)
            for i in range(len(unique_pairs)):
                transfer[i, unique_inv == i] = 1

            self.weights = transfer @ self.weights

            # NB update self.endog, self.exposure, self.exog, self.weights for unique_pairs compression:
            self.endog = unique_pairs[:, 0]
            self.exposure = unique_pairs[:, 1]

            # NB one-hot encoded design matrix of class labels
            self.exog = self.exog[unique_idx, :]

    def nloglikeobs(self, params):
        p = self.exog @ params[:-1]

        if self.tumor_prop is None:
            a = p * params[-1]
            b = (1.0 - p) * params[-1]
        else:
            a = (p * self.tumor_prop + 0.5 * (1.0 - self.tumor_prop)) * params[-1]
            b = ((1.0 - p) * self.tumor_prop + 0.5 * (1.0 - self.tumor_prop)) * params[
                -1
            ]

        return -scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b).dot(
            self.weights
        )

    def fit(self, start_params=None, maxiter=10_000, maxfun=5_000, **kwargs):
        using_default_params = start_params is None
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                start_params = np.append(
                    0.5 / self.exog.shape[1] * np.ones(self.nparams), 1.0
                )

        start_time = time.time()

        # NB log initial start params and initial likelihood:
        logger.info(f"Weighted_BetaBinom_mix initial likelihood={self.nloglikeobs(start_params):.6e} @ start_params: {start_params}")
        
        exit(0)

        result = super().fit(
            start_params=start_params,
            maxiter=maxiter,
            maxfun=maxfun,
            method=get_solver(),
            **kwargs,
        )
        runtime = time.time() - start_time

        logger.debug(
            f"Weighted_BetaBinom_mix debug - mle_retvals: {result.mle_retvals}, "
            f"mle_settings: {result.mle_settings}"
        )

        logger.info(
            f"Weighted_BetaBinom_mix done: {runtime:.2f}s with tumor_prop={self.tumor_prop is not None}, "
            f"{len(start_params)} params ({'with default start' if using_default_params else 'with custom start'}), "
            f"{result.mle_retvals.get('iterations', 'N/A')} iter, "
            f"{result.mle_retvals.get('fcalls', 'N/A')} fcalls, "
            f"optimizer: {result.mle_settings.get('optimizer', 'Unknown')}, "
            f"converged: {result.mle_retvals.get('converged', 'N/A')}, "
            f"llf: {result.llf:.6e}"
        )

        return result


# LEGACY
Weighted_NegativeBinomial = partial(Weighted_NegativeBinomial_mix, tumor_prop=None)
Weighted_BetaBinom = partial(Weighted_BetaBinom_mix, tumor_prop=None)
