import time
import warnings

import numpy as np
import scipy.stats
from math import lgamma
from scipy.special import loggamma
from functools import partial
from cnaster.config import get_global_config
from cnaster.hmm_utils import get_solver

# from cnaster.hmm_mcmc import run_mcmc_numba, plot_mcmc, numba_nloglikeobs_nb, numba_nloglikeobs_bb
from dataclasses import dataclass, asdict
from typing import Optional, Any
import csv
from pathlib import Path
from numba import njit
import scipy.optimize
from cnaster.config import start_time
from cnaster.logger import get_logger


logger = get_logger(__name__, start_time=start_time)

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

"""
def nloglikeobs_nb(
    endog,
    exog,
    weights,
    exposure,
    params,
    reduce=True,
):
    num_states = exog.shape[-1]
    nb_mean = exog @ np.exp(params[:num_states]) * exposure
    nb_disp = exog @ params[num_states:]

    # NB vectorized call.
    n, p = convert_params_disp(nb_mean, nb_disp)

    result = -scipy.stats.nbinom.logpmf(endog, n, p)
    result[np.isnan(result)] = np.inf

    if reduce:
        result = result.dot(weights)
        assert not np.isnan(result), f"{params}: {result}"

    return result
"""

def betabinom_logpmf_zp(endog, exposure):
    return loggamma(exposure + 1) - loggamma(endog + 1) - loggamma(exposure - endog + 1)


@njit(nogil=True, cache=True, error_model="numpy")
def compute_bb_ab(exog, params):
    num_states = exog.shape[-1]

    p = np.dot(exog, params[:num_states])
    t = np.dot(exog, params[num_states:])

    a = p * t
    b = (1.0 - p) * t

    return a, b


@njit(nogil=True, cache=True, error_model="numpy")
def betabinom_logpmf(endog, exposure, a, b, zero_point, EPS=1.0e-10):
    result_array = np.empty_like(endog, dtype=np.float64)

    for i in range(len(endog)):
        ai = a[i]
        bi = b[i]

        # NB guard against numerical instability at 0
        if ai < EPS:
            ai = EPS
        if bi < EPS:
            bi = EPS

        result_array[i] = (
            zero_point[i]
            + lgamma(endog[i] + ai)
            + lgamma(exposure[i] - endog[i] + bi)
            + lgamma(ai + bi)
            - lgamma(exposure[i] + ai + bi)
            - lgamma(ai)
            - lgamma(bi)
        )
        if np.isnan(result_array[i]):
            result_array[i] = -np.inf

    return result_array


def nloglikeobs_bb(
    endog,
    exog,
    weights,
    exposure,
    params,
    zero_point=None,
    reduce=True,
):
    a, b = compute_bb_ab(exog, params)

    if zero_point is not None:
        result = -betabinom_logpmf(endog, exposure, a, b, zero_point)
    else:
        result = -scipy.stats.betabinom.logpmf(endog, exposure, a, b)
        result[np.isnan(result)] = np.inf

    if reduce:
        reduced_result = result.dot(weights)

        if np.isnan(reduced_result):
            logger.info(
                f"Detected invalid ln. likelihood={reduced_result} for:\n{params}"
            )

            nan_mask = np.isnan(weights)
            nan_weights = weights[nan_mask]

            nan_mask = np.isnan(result)
            nan_endog = np.unique(endog[nan_mask])
            nan_exposure = np.unique(exposure[nan_mask])
            nan_alphas = np.unique(a[nan_mask])
            nan_betas = np.unique(b[nan_mask])

            logger.info(
                f"NaN identified:\n"
                f"  weights: {nan_weights}\n"
                f"  endog: {nan_endog}\n"
                f"  exposure: {nan_exposure}\n"
                f"  alphas: {nan_alphas}\n"
                f"  betas: {nan_betas}\n"
                f"  Fraction of NaN observations: {np.mean(nan_mask):.6e}"
            )

            raise RuntimeError()

        result = reduced_result

    return result

"""
class Weighted_NegativeBinomial_mix:
    def __init__(
        self,
        endog,
        exog,
        weights,
        exposure,
        fixed_dispersion=False,
        shared_dispersion=True,
        max_rdr=5.0,  # TODO HACK MAGIC
    ):
        exog = exog.copy()

        # NB corresponds to a single (potentially unknown) state.
        if exog.ndim == 1:
            exog = np.atleast_2d(exog).T

        # NB EM-based posterior weights.
        self.endog = np.asarray(endog, dtype=np.float64)
        self.exog = np.asarray(exog, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.exposure = np.asarray(exposure, dtype=np.float64)
        self.fixed_dispersion = fixed_dispersion
        self.shared_dispersion = shared_dispersion

        self.num_states = self.exog.shape[-1]

        # TODO HACK
        if np.mean(exposure == 0.0) > 0.0:
            logger.warning(
                f"Removing posterior weight of {100. * np.mean(exposure == 0.0):.3f} [%] data with zero exposure."
            )

            # NB equivalent to dropping data; weights are not otherwise renormalized.
            self.weights[exposure == 0.0] = 0.0

        # TODO HACK
        if max_rdr is not None:
            logger.warning(
                f"Assuming max_rdr={max_rdr}, which removes {100. * np.mean(endog > max_rdr * exposure):.3f} [%]"
            )

            # NB equivalent to dropping data; weights are not otherwise renormalized.
            self.weights[endog > max_rdr * exposure] = 0.0

    def nloglikeobs(self, params, *args):
        params = np.array(params)

        if self.fixed_dispersion:
            params = np.concatenate([params, np.full(self.num_states, args[0])])
        elif self.shared_dispersion:
            params = np.concatenate([params[:-1], np.full(self.num_states, params[-1])])
        else:
            assert len(params) == self.num_states * 2

        return nloglikeobs_nb(
            self.endog,
            self.exog,
            self.weights,
            self.exposure,
            params,
            reduce=True,
        )

    def get_default_params(self, legacy=False):
        ms, single_dispersion = get_nbinom_start_params(legacy=legacy)

        if self.fixed_dispersion or self.shared_dispersion:
            disp = [single_dispersion]
        else:
            disp = [single_dispersion] * self.num_states

        return np.array(ms[: self.num_states] + disp)

    def get_bounds(self):
        EPSILON = 1.0e-6
        bounds = []

        # NB bounds for log-space parameters (can be negative)
        for _ in range(self.num_states):
            bounds.append((-10, 10))

        # NB bound for overdispersion parameter (must be positive)
        if self.fixed_dispersion:
            pass
        elif self.shared_dispersion:
            bounds.append((EPSILON, 1e6))
        else:
            for _ in range(self.num_states):
                bounds.append((EPSILON, 1e6))

        return bounds

    def fit(
        self,
        start_params=None,
        maxiter=10_000,
        maxfun=5_000,
        legacy=False,
        **kwargs,
    ):
        if using_default_params := (start_params is None):
            start_params = self.get_default_params(legacy=legacy)

        start_time = time.time()

        logger.info(
            f"Weighted_NegativeBinomial_mix (num_states={self.num_states}, endog_shape={self.endog.shape}) initial -ln likelihood={self.nloglikeobs(start_params):.6e} @ start_params:\n{[xx for xx in start_params]}"
        )

        bounds = self.get_bounds()
        options = {
            "maxiter": maxiter,
            "maxfun": maxfun,
            "ftol": kwargs.get("ftol", None),
            "disp": kwargs.get("disp", False),
        }

        if self.fixed_dispersion:
            args = (start_params[-1],)
            start_params = start_params[:-1]
        else:
            args = ()

        result = scipy.optimize.minimize(
            self.nloglikeobs,
            start_params,
            method=get_solver(),
            bounds=bounds,
            options=options,
            args=args,
        )

        optimize_result = OptimizationResult(
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
            optimize_result,
        )

        logger.info(
            f"Weighted_NegativeBinomial_mix done: {runtime:.2f}s with\nendog_shape={self.endog.shape},\n"
            f"{len(start_params)} params ({'with default start' if using_default_params else 'with custom start'}),\n"
            f"{optimize_result.mle_retvals.get('iterations', 'N/A')} iter,\n"
            f"{optimize_result.mle_retvals.get('fcalls', 'N/A')} fcalls,\n"
            f"optimizer: {optimize_result.mle_settings.get('optimizer', 'Unknown')},\n"
            f"converged: {optimize_result.mle_retvals.get('converged', 'N/A')},\n"
            f"llf: {optimize_result.llf:.6e}\n"
            f"params:\n{[xx for xx in optimize_result.params]}"
        )

        return optimize_result
"""

class Weighted_BetaBinom_mix:
    def __init__(
        self,
        endog,
        exog,
        weights,
        exposure,
        tumor_prop=None,
        fixed_dispersion=False,
        shared_dispersion=True,
    ):
        exog = exog.copy()

        # NB corresponds to a single (potentially unknown) state.
        if exog.ndim == 1:
            exog = np.atleast_2d(exog).T

        # NB EM-based posterior weights.
        self.endog = np.asarray(endog, dtype=np.float64)
        self.exog = np.asarray(exog, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.exposure = np.asarray(exposure, dtype=np.float64)
        self.tumor_prop = tumor_prop
        self.fixed_dispersion = fixed_dispersion
        self.shared_dispersion = shared_dispersion

        self.num_states = self.exog.shape[-1]
        self.zero_point = None

    def nloglikeobs(self, params, *args):
        params = np.array(params)

        if self.fixed_dispersion:
            params = np.concatenate([params, np.full(self.num_states, args[0])])
        elif self.shared_dispersion:
            params = np.concatenate([params[:-1], np.full(self.num_states, params[-1])])
        else:
            assert len(params) == self.num_states * 2

        return nloglikeobs_bb(
            self.endog,
            self.exog,
            self.weights,
            self.exposure,
            params,
            zero_point=self.zero_point,
            reduce=True,
        )

    def get_default_params(self, legacy=False):
        ps, single_dispersion = get_betabinom_start_params(
            legacy=legacy, exog=self.exog
        )

        if self.fixed_dispersion or self.shared_dispersion:
            disp = [single_dispersion]
        else:
            disp = [single_dispersion] * self.num_states

        return np.array(ps[: self.num_states] + disp)

    def get_bounds(self):
        EPSILON = 1.0e-6
        bounds = []

        for _ in range(self.num_states):
            bounds.append((EPSILON, 1.0 - EPSILON))

        if self.fixed_dispersion:
            pass
        elif self.shared_dispersion:
            bounds.append((EPSILON, 1e6))
        else:
            for _ in range(self.num_states):
                bounds.append((EPSILON, 1e6))

        return bounds

    def fit(
        self, start_params=None, maxiter=10_000, maxfun=5_000, legacy=False, **kwargs
    ):
        if using_default_params := (start_params is None):
            start_params = self.get_default_params(legacy=legacy)

        self.zero_point = betabinom_logpmf_zp(self.endog, self.exposure)

        start_time = time.time()

        logger.info(
            f"Weighted_BetaBinom_mix (num_states={self.num_states}, endog.shape={self.endog.shape}), initial nloglike={self.nloglikeobs(start_params):.6e} @ start_params:\n{[xx for xx in start_params]}"
        )

        bounds = self.get_bounds()
        options = {
            "maxiter": maxiter,
            "maxfun": maxfun,
            "ftol": kwargs.get("ftol", None),
            "disp": kwargs.get("disp", False),
        }

        if self.fixed_dispersion:
            args = (start_params[-1],)
            start_params = start_params[:-1]
        else:
            args = ()

        result = scipy.optimize.minimize(
            self.nloglikeobs,
            start_params,
            method=get_solver(),
            bounds=bounds,
            options=options,
            args=args,
        )

        optimize_result = OptimizationResult(
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
            optimize_result,
        )

        logger.info(
            f"Weighted_BetaBinom_mix done: {runtime:.2f}s with {get_solver()}\nendog_shape={self.endog.shape},\ntumor_prop={self.tumor_prop is not None},\n"
            f"{len(start_params)} params ({'with default start' if using_default_params else 'with custom start'}),\n"
            f"{optimize_result.mle_retvals.get('iterations', 'N/A')} iter,\n"
            f"{optimize_result.mle_retvals.get('fcalls', 'N/A')} fcalls,\n"
            f"optimizer: {optimize_result.mle_settings.get('optimizer', 'Unknown')},\n"
            f"converged: {optimize_result.mle_retvals.get('converged', 'N/A')},\n"
            f"llf: {optimize_result.llf:.6e}\n"
            f"params:\n{[xx for xx in optimize_result.params]}"
        )

        return optimize_result


# LEGACY
# Weighted_NegativeBinomial = partial(Weighted_NegativeBinomial_mix)
Weighted_BetaBinom = partial(Weighted_BetaBinom_mix)
