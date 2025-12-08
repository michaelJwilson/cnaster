import logging
import time
import warnings

import numpy as np
import scipy.stats
from math import lgamma
from scipy.special import loggamma
from functools import partial
from cnaster.config import get_global_config
from cnaster.hmm_utils import convert_params_disp, get_solver
from cnaster.priors import baf_prior_eval, rdr_prior_eval
from dataclasses import dataclass, asdict
from typing import Optional, Any
import csv
from pathlib import Path
from numba import njit
import scipy.optimize

logger = logging.getLogger(__name__)

# TODO
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


@dataclass
class OptimizationResult:
    optimizer: str
    params: np.ndarray
    llf: float
    converged: bool
    iterations: int
    fcalls: int

    def __post_init__(self):
        if not hasattr(self, "mle_retvals"):
            self.mle_retvals = {
                "converged": self.converged,
                "iterations": self.iterations,
                "fcalls": self.fcalls,
            }
        if not hasattr(self, "mle_settings"):
            self.mle_settings = {"optimizer": self.optimizer}


@dataclass
class FitMetrics:
    row_index: int
    timestamp: str
    model: str
    optimizer: str
    size: int
    num_states: int
    default_start_params: bool
    runtime: str
    iterations: Optional[int]
    fcalls: Optional[int]
    converged: Optional[bool]
    llf: float


def get_nbinom_start_params(legacy=False, jitter=False):
    config = get_global_config()

    if legacy:
        return 0.1 * np.ones(config.hmm.n_states), 1.0e-2

    ms = [float(xx) for xx in config.nbinom.start_params.split(",")]

    if jitter:
        ms = [mm + 1.0e-2 * np.random.rand() for mm in ms]

    return ms, float(config.nbinom.start_disp)


def get_betabinom_start_params(legacy=False, exog=None):
    config = get_global_config()

    if legacy:
        return (0.5 / exog.shape[1]) * np.ones(config.hmm.n_states), 1.0

    ps = [float(xx) for xx in config.betabinom.start_params.split(",")]

    return ps, float(config.betabinom.start_disp)


def flush_perf(
    model: str,
    size,
    num_states: int,
    default_start_params: bool,
    start_time: float,
    end_time: float,
    result: Any,
):
    config = get_global_config()
    runtime = end_time - start_time

    mle_retvals = getattr(result, "mle_retvals", {})
    mle_settings = getattr(result, "mle_settings", {})

    # Read existing file to get next row index
    perf_file = Path(config.paths.perf_path)
    row_index = 1
    if perf_file.exists():
        try:
            with open(perf_file, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                row_index = sum(1 for _ in reader)  # Count all rows including header
        except:
            row_index = 1

    metrics = FitMetrics(
        row_index=row_index,
        model=model,
        runtime=f"{runtime:.4f}",
        iterations=mle_retvals.get("iterations"),
        fcalls=mle_retvals.get("fcalls"),
        optimizer=mle_settings.get("optimizer", "Unknown").ljust(40),
        size=size,
        num_states=num_states,
        default_start_params=default_start_params,
        converged=mle_retvals.get("converged"),
        llf=f"{result.llf:.6e}" if hasattr(result, "llf") else "NAN",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    file_exists = perf_file.exists()

    with open("cnaster.perf", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(metrics).keys(), delimiter="\t")

        if not file_exists:
            writer.writeheader()

        writer.writerow(asdict(metrics))


@njit
def collapse_exog(exog):
    num_obs, num_states = exog.shape
    states = np.arange(num_states)
    switches = np.zeros(num_states)

    current_state = states[0]

    for ii in range(num_obs):
        for jj in range(num_states):
            if exog[ii, jj] == 1:
                if jj != current_state:
                    switches[current_state] = ii
                    current_state = jj

                break

    switches[-1] = num_obs

    return states, switches


def nloglikeobs_nb(
    endog,
    exog,
    weights,
    exposure,
    params,
    tumor_prop=None,
    prior=False,
    reduce=True,
):
    if tumor_prop is None:
        nb_mean = exog @ np.exp(params[:-1]) * exposure
    else:
        nb_mean = exposure * (
            tumor_prop * exog @ np.exp(params[:-1]) + (1.0 - tumor_prop)
        )

    nb_std = np.sqrt(nb_mean + params[-1] * nb_mean**2)
    n, p = convert_params_disp(nb_mean, params[-1])

    result = -scipy.stats.nbinom.logpmf(endog, n, p)
    result[np.isnan(result)] = np.inf

    if prior:
        # TODO tumor prop
        result -= rdr_prior_eval(exog @ np.exp(params[:-1]), sigma=None)

    if reduce:
        result = result.dot(weights)
        assert not np.isnan(result), f"{params}: {result}"

    return result


def betabinom_logpmf_zp(endog, exposure):
    return loggamma(exposure + 1) - loggamma(endog + 1) - loggamma(exposure - endog + 1)


@njit(nogil=True, cache=True, fastmath=False, error_model="numpy")
def compute_bb_ab(exog, params, tumor_prop=None):
    p = np.dot(exog, params[:-1])
    tau = params[-1]

    if tumor_prop is None:
        a = p * tau
        b = (1.0 - p) * tau
    else:
        a = (p * tumor_prop + 0.5 * (1.0 - tumor_prop)) * tau
        b = ((1.0 - p) * tumor_prop + 0.5 * (1.0 - tumor_prop)) * tau

    return a, b


@njit(nogil=True, cache=True, fastmath=False, error_model="numpy")
def betabinom_logpmf(endog, exposure, a, b, zero_point):
    result_array = np.empty_like(endog, dtype=np.float64)

    for i in range(len(endog)):
        result_array[i] = (
            zero_point[i]
            + lgamma(endog[i] + a[i])
            + lgamma(exposure[i] - endog[i] + b[i])
            + lgamma(a[i] + b[i])
            - lgamma(exposure[i] + a[i] + b[i])
            - lgamma(a[i])
            - lgamma(b[i])
        )
        if np.isnan(result_array[i]):
            result_array[i] = np.inf

    return result_array


def nloglikeobs_bb(
    endog,
    exog,
    weights,
    exposure,
    params,
    tumor_prop=None,
    zero_point=None,
    reduce=True,
    prior=False,
):
    a, b = compute_bb_ab(exog, params, tumor_prop)

    if zero_point is not None:
        result = -betabinom_logpmf(endog, exposure, a, b, zero_point)
    else:
        result = -scipy.stats.betabinom.logpmf(endog, exposure, a, b)
        result[np.isnan(result)] = np.inf

    # if prior:
    #   prior_shift = baf_prior_eval(a / (a + b), sigma=None)
    #   result -= prior_shift

    if reduce:
        reduced_result = result.dot(weights)

        if np.isnan(reduced_result):
            logger.info(
                f"Detected invalid ln. likelihood={reduced_result} for:\n{params}"
            )

            nan_mask = np.isnan(result)
            nan_endog = np.unique(endog[nan_mask])
            nan_exposure = np.unique(exposure[nan_mask])
            nan_alphas = np.unique(a[nan_mask])
            nan_betas = np.unique(b[nan_mask])

            logger.info(
                f"NaN identified:\n"
                f"  endog: {nan_endog}\n"
                f"  exposure: {nan_exposure}\n"
                f"  alphas: {nan_alphas}\n"
                f"  betas: {nan_betas}\n"
                f"  Fraction of NaN observations: {np.mean(nan_mask):.6e}"
            )

            raise RuntimeError()

        result = reduced_result

    return result


class Weighted_NegativeBinomial_mix:
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
        self,
        endog,
        exog,
        weights,
        exposure,
        tumor_prop=None,
        compress=True,
        seed=0,
        max_rdr=5.0,  # TODO HACK MAGIC
        **kwargs,
    ):
        exog = exog.copy()

        if exog.ndim == 1:
            exog = np.atleast_2d(exog).T

        # NB EM-based posterior weights.
        self.endog = np.asarray(endog, dtype=np.float64)
        self.exog = np.asarray(exog, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.exposure = np.asarray(exposure, dtype=np.float64)
        self.seed = seed
        self.tumor_prop = tumor_prop
        self.compress = False
        self.num_states = self.exog.shape[-1]

        # TODO HACK
        if np.mean(exposure == 0.0) > 0.0:
            logger.warning(
                f"Removing posterior weight of {100. * np.mean(exposure == 0.0):.3f} [%] data with zero exposure."
            )

            # TODO HACK renormalize?
            self.weights[exposure == 0.0] = 0.0

        # TODO HACK
        if max_rdr is not None:
            logger.warning(
                f"Assuming max_rdr={max_rdr}, which removes {100. * np.mean(endog > max_rdr * exposure):.3f} [%]"
            )

            # TODO HACK renormalize?
            self.weights[endog > max_rdr * exposure] = 0.0

        if tumor_prop is not None:
            logger.warning(
                f"{self.__class__.__name__} compression is not supported for tumor_prop != None."
            )
            return

        # NB instigate compression of likelihood: do not repeat evaluation of probabilities given
        #    'same' counts (up to decimals definition below).
        cls = np.argmax(self.exog, axis=-1)
        counts = np.vstack([self.endog, self.exposure, cls]).T

        # TODO HACK decimals
        if counts.dtype != int:
            counts = counts.round(decimals=4)

        # NB see https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        unique_pairs, unique_idx, unique_inv = np.unique(
            counts, return_index=True, return_inverse=True, axis=0
        )

        mean_compression = 1.0 - len(unique_pairs) / len(self.endog)

        logger.warning(
            f"{self.__class__.__name__} has further achievable compression: {100. * mean_compression:.4f}%"
        )

        if compress and mean_compression > 0.0:
            transfer = np.zeros((len(unique_pairs), len(self.endog)), dtype=int)

            for i in range(len(unique_pairs)):
                transfer[i, unique_inv == i] = 1

            self.weights = transfer @ self.weights

            self.endog = unique_pairs[:, 0]
            self.exposure = unique_pairs[:, 1]

            self.exog = self.exog[unique_idx, :]
            self.compress = True

    def nloglikeobs(self, params, reduce=True):
        return nloglikeobs_nb(
            self.endog,
            self.exog,
            self.weights,
            self.exposure,
            params,
            tumor_prop=self.tumor_prop,
            reduce=reduce,
        )

    def get_bounds(self, params):
        """
        Set reasonable bounds for parameters
        """
        n_params = len(params)
        bounds = []

        EPSILON = 1.0e-6

        # NB bounds for log-space parameters (can be negative)
        for _ in range(n_params - 1):
            bounds.append((-10, 10))

        # NB bound for overdispersion parameter (must be positive)
        bounds.append((EPSILON, 1e6))

        return bounds

    def fit(
        self,
        start_params=None,
        maxiter=10_000,
        maxfun=5_000,
        legacy=False,
        clip_percentile=None,
        empirical=False,
        **kwargs,
    ):
        using_default_params = start_params is None

        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                ms, disp = get_nbinom_start_params(legacy=legacy)
                start_params = np.array(ms[: self.num_states] + [disp])

        assert self.num_states == (
            len(start_params) - 1
        ), f"{len(start_params)}, {self.exog.shape}"

        state_assignment = np.argmax(self.exog, axis=-1)

        # TODO? weighted median?
        empirical_rdr_mean = np.nan * np.ones(self.num_states)
        empirical_rdr_std = np.nan * np.ones(self.num_states)

        for state_idx in range(self.num_states):
            state_mask = state_assignment == state_idx
            if not np.any(state_mask):
                logger.warning(f"State {state_idx} has no observations, using defaults")
                continue

            state_rdr = self.endog[state_mask] / np.maximum(
                self.exposure[state_mask], 1.0e-10
            )
            state_weights = self.weights[state_mask]

            total_weight = np.sum(state_weights)

            if total_weight > 0.0:
                empirical_rdr_mean[state_idx] = (
                    np.sum(state_rdr * state_weights) / total_weight
                )
                empirical_rdr_std[state_idx] = np.sqrt(
                    np.sum(
                        state_weights * (state_rdr - empirical_rdr_mean[state_idx]) ** 2
                    )
                    / total_weight
                )

        start_time = time.time()

        logger.info(
            f"Weighted_NegativeBinomial_mix (compress={self.compress}, num_states={self.num_states}, endog_shape={self.endog.shape}) initial -ln likelihood={self.nloglikeobs(start_params):.6e} @ start_params:\n{[xx for xx in start_params]}"
        )

        bounds = self.get_bounds(start_params)
        options = {
            "maxiter": maxiter,
            "maxfun": maxfun,
            "ftol": kwargs.get("ftol", None),
            "disp": kwargs.get("disp", False),
        }

        if clip_percentile is not None and not using_default_params:
            obs_nll = nloglikeobs_nb(
                self.endog,
                self.exog,
                self.weights,
                self.exposure,
                start_params,
                tumor_prop=self.tumor_prop,
                reduce=False,
            )

            state_assignment = np.argmax(self.exog, axis=-1)
            total_clipped = 0

            for state_idx in range(self.num_states):
                state_mask = state_assignment == state_idx
                state_weight = np.sum(self.weights[state_mask])

                state_nlls = obs_nll[state_mask]

                threshold = np.percentile(state_nlls, clip_percentile)

                clip_mask = state_mask & (obs_nll > threshold)
                n_clipped = np.sum(clip_mask)

                # TODO HACK self.weights renormalization? self.clipped_weights ...
                self.weights[clip_mask] = 0.0

                clipped_state_weights = np.sum(self.weights[state_mask])
                total_clipped += n_clipped

                logger.info(
                    f"Clipped state {state_idx} with weight {state_weight:.2f} to {100. * clipped_state_weights / state_weight:.2f} [%] of original."
                )

            logger.warning(
                f"Clipped @ {clip_percentile}[%] percentile with -ln likelihood={self.nloglikeobs(start_params):.6e} @ start_params:\n{[xx for xx in start_params]}"
            )

        state_weights = []

        for state_idx in range(self.num_states):
            state_mask = state_assignment == state_idx
            state_weight = np.sum(self.weights[state_mask])

            state_weights.append(state_weight)

        state_weights = np.array(state_weights)

        empirical_log_mu = np.log(empirical_rdr_mean)

        empirical_disp_state = (
            np.maximum(empirical_rdr_std**2.0, empirical_rdr_mean) - empirical_rdr_mean
        ) / (empirical_rdr_mean**2.0)
        empirical_disp = np.sum(empirical_disp_state * state_weights) / total_weight

        if empirical:
            params = np.array(list(empirical_log_mu) + [empirical_disp])
            result = OptimizationResult(
                optimizer="empirical",
                params=params,
                llf=-self.nloglikeobs(params),
                converged=1,
                iterations=None,
                fcalls=None,
            )

            if ~np.isfinite(empirical_disp) or empirical_disp <= 0.0:
                # TODO HACK nb emission for disp=0.0
                empirical_disp = 1.0e-6
                logger.warning(
                    f"Empirical dispersion is non-finite or non-positive: {empirical_disp_state}, assigning {empirical_disp}."
                )

            logger.info(
                "Empirical RDR per state:\n"
                + "\n".join(
                    [
                        f"  State {i}: mean={empirical_rdr_mean[i]:.4f}\tstd={empirical_rdr_std[i]:.4f}"
                        for i in range(self.num_states)
                    ]
                )
            )

            return result

        result = scipy.optimize.minimize(
            self.nloglikeobs,
            start_params,
            method=get_solver(),
            bounds=bounds,
            options=options,
        )

        result = OptimizationResult(
            optimizer=get_solver(),
            params=result.x,
            llf=-result.fun,
            converged=result.success,
            iterations=result.get("nit", None),
            fcalls=result.get("nfev", None),
        )

        end_time = time.time()
        runtime = end_time - start_time

        flush_perf(
            self.__class__.__name__,
            len(self.exog),
            self.num_states,
            using_default_params,
            start_time,
            end_time,
            result,
        )

        logger.debug(
            f"Weighted_NegativeBinomial_mix debug - mle_retvals: {result.mle_retvals}, "
            f"mle_settings: {result.mle_settings}"
        )

        logger.info(
            f"Weighted_NegativeBinomial_mix done: {runtime:.2f}s with\nendog_shape={self.endog.shape},\ntumor_prop={self.tumor_prop is not None},\n"
            f"{len(start_params)} params ({'with default start' if using_default_params else 'with custom start'}),\n"
            f"{result.mle_retvals.get('iterations', 'N/A')} iter,\n"
            f"{result.mle_retvals.get('fcalls', 'N/A')} fcalls,\n"
            f"optimizer: {result.mle_settings.get('optimizer', 'Unknown')},\n"
            f"converged: {result.mle_retvals.get('converged', 'N/A')},\n"
            f"llf: {result.llf:.6e}\n"
            f"params: {result.params}"
        )

        fitted_rdr_mean = np.exp(result.params[:-1])
        fitted_disp = result.params[-1]
        predicted_std = np.sqrt(fitted_rdr_mean + fitted_disp * fitted_rdr_mean**2)

        logger.info(
            "Empirical & best-fit RDR per state:\n"
            + "\n".join(
                [
                    f"  State {i}: emp. mean={empirical_rdr_mean[i]:.4f}\t fit mean={fitted_rdr_mean[i]:.4f}\t emp. std={empirical_rdr_std[i]:.4f}\t fit std={predicted_std[i]:.4f}"
                    for i in range(self.num_states)
                ]
            )
        )

        return result


class Weighted_BetaBinom_mix:
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
        exog = exog.copy()

        if exog.ndim == 1:
            exog = np.atleast_2d(exog).T

        # NB EM-based posterior weights.
        self.endog = np.asarray(endog, dtype=np.float64)
        self.exog = np.asarray(exog, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.exposure = np.asarray(exposure, dtype=np.float64)
        self.tumor_prop = tumor_prop
        self.compress = False
        self.num_states = self.exog.shape[-1]
        self.zero_point = None

        if tumor_prop is not None:
            logger.warning(
                f"{self.__class__.__name__} compression is not supported for tumor_prop != None."
            )
            return

        # TODO HACK
        cls = np.argmax(self.exog, axis=-1)
        counts = np.vstack([self.endog, self.exposure, cls]).T

        # TODO HACK decimals
        if counts.dtype != int:
            counts = counts.round(decimals=4)

        # NB see https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        unique_pairs, unique_idx, unique_inv = np.unique(
            counts, return_index=True, return_inverse=True, axis=0
        )

        mean_compression = 1.0 - len(unique_pairs) / len(self.endog)

        logger.warning(
            f"{self.__class__.__name__} has further achievable compression: {100. * mean_compression:.4f}%"
        )

        if compress and mean_compression > 0.0:
            # TODO HACK
            # NB sum self.weights - relies on original self.endog length
            transfer = np.zeros((len(unique_pairs), len(self.endog)), dtype=int)

            for i in range(len(unique_pairs)):
                transfer[i, unique_inv == i] = 1

            self.weights = transfer @ self.weights

            # NB update self.endog, self.exposure, self.exog, self.weights for unique_pairs compression:
            self.endog = unique_pairs[:, 0]
            self.exposure = unique_pairs[:, 1]

            # NB one-hot encoded design matrix of class labels
            self.exog = self.exog[unique_idx, :]
            self.compress = True

    def nloglikeobs(self, params, reduce=True):
        return nloglikeobs_bb(
            self.endog,
            self.exog,
            self.weights,
            self.exposure,
            params,
            tumor_prop=self.tumor_prop,
            zero_point=self.zero_point,
            reduce=reduce,
        )

    def get_bounds(self, params):
        """
        Set reasonable bounds for parameters
        """
        n_params = len(params)
        bounds = []

        EPSILON = 1.0e-6

        for i in range(n_params - 1):
            bounds.append((EPSILON, 1.0 - EPSILON))

        bounds.append((EPSILON, 1e6))

        return bounds

    def fit(
        self, start_params=None, maxiter=10_000, maxfun=5_000, legacy=False, **kwargs
    ):
        using_default_params = start_params is None

        if start_params is None:
            # TODO DEPRECATE
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                ps, disp = get_betabinom_start_params(legacy=legacy, exog=self.exog)
                start_params = np.array(ps[: self.num_states] + [disp])

        assert self.num_states == (
            len(start_params) - 1
        ), f"{len(start_params)}, {self.exog.shape}"

        self.zero_point = betabinom_logpmf_zp(self.endog, self.exposure)

        start_time = time.time()

        logger.info(
            f"Weighted_BetaBinom_mix (compress={self.compress}, num_states={self.num_states}, endog_shape={self.endog.shape}) initial likelihood={self.nloglikeobs(start_params):.6e} @ start_params:\n{[xx for xx in start_params]}"
        )

        bounds = self.get_bounds(start_params)
        options = {
            "maxiter": maxiter,
            "maxfun": maxfun,
            "ftol": kwargs.get("ftol", None),
            "disp": kwargs.get("disp", False),
        }

        result = scipy.optimize.minimize(
            self.nloglikeobs,
            start_params,
            method=get_solver(),
            bounds=bounds,
            options=options,
        )

        # TODO
        # if "hess_inv" in result:
        #     hess_inv = result.hess_inv.todense()
        #     sigmas = np.sqrt(np.maximum(np.diag(hess_inv), 0.0))
        #     print(f"sigmas=\n{sigmas}")

        result = OptimizationResult(
            optimizer=get_solver(),
            params=result.x,
            llf=-result.fun,
            converged=result.success,
            iterations=result.get("nit", None),
            fcalls=result.get("nfev", None),
        )

        end_time = time.time()
        runtime = end_time - start_time

        flush_perf(
            self.__class__.__name__,
            len(self.exog),
            self.num_states,
            using_default_params,
            start_time,
            end_time,
            result,
        )

        logger.debug(
            f"Weighted_BetaBinom_mix debug - mle_retvals: {result.mle_retvals}, "
            f"mle_settings: {result.mle_settings}"
        )

        logger.info(
            f"Weighted_BetaBinom_mix done: {runtime:.2f}s with {get_solver()}\nendog_shape={self.endog.shape},\ntumor_prop={self.tumor_prop is not None},\n"
            f"{len(start_params)} params ({'with default start' if using_default_params else 'with custom start'}),\n"
            f"{result.mle_retvals.get('iterations', 'N/A')} iter,\n"
            f"{result.mle_retvals.get('fcalls', 'N/A')} fcalls,\n"
            f"optimizer: {result.mle_settings.get('optimizer', 'Unknown')},\n"
            f"converged: {result.mle_retvals.get('converged', 'N/A')},\n"
            f"llf: {result.llf:.6e}\n"
            f"params:\n{[xx for xx in result.params]}"
        )

        return result


# LEGACY
Weighted_NegativeBinomial = partial(Weighted_NegativeBinomial_mix, tumor_prop=None)
Weighted_BetaBinom = partial(Weighted_BetaBinom_mix, tumor_prop=None)
