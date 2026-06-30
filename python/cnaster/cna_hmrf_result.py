import copy
import numpy as np
import scipy.special
from typing import Optional, Any
from dataclasses import dataclass, fields
from cnaster.hmm_initialize import gmm_init
from cnaster.hmm_sitewise import hmm_sitewise
from cnaster.hmm_utils import compute_posterior_obs
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


@dataclass
class HMMParams:
    new_log_mu: np.ndarray
    new_alphas: np.ndarray
    new_p_binom: np.ndarray
    new_taus: np.ndarray
    new_log_startprob: np.ndarray
    new_log_transmat: np.ndarray


@dataclass
class HMMProfile:
    log_gamma: np.ndarray
    pred_cnv: np.ndarray


@dataclass
class CloneAssignment:
    assignment_before_reindex: Optional[np.ndarray] = None
    prev_assignment: Optional[np.ndarray] = None
    new_assignment: Optional[np.ndarray] = None
    total_llf: float = np.nan


@dataclass
class CnaHMRFResult:
    params: HMMParams
    profile: HMMProfile
    llf: float
    n_states: int
    assignment: CloneAssignment

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)

        if hasattr(self.params, key):
            return getattr(self.params, key)

        if hasattr(self.profile, key):
            return getattr(self.profile, key)

        if hasattr(self.assignment, key):
            return getattr(self.assignment, key)

        raise KeyError(
            f"'{key}' not found in CnaHMRFResult, HMMParams, HMMProfile, or CloneAssignment."
        )

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        elif hasattr(self.params, key):
            setattr(self.params, key, value)
        elif hasattr(self.profile, key):
            setattr(self.profile, key, value)
        elif hasattr(self.assignment, key):
            setattr(self.assignment, key, value)
        else:
            raise KeyError(f"Cannot set unknown key '{key}'")

    # NB mirror shallow copy behavior a dictionary.
    def copy(self, deep: bool = False) -> "CnaHMRFResult":
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def keys(self):
        all_keys = []

        # NB add top-level scalars (llf, n_states)
        for f in fields(self):
            if f.name not in {"params", "profile", "assignment"}:
                all_keys.append(f.name)

        for sub_obj in [self.params, self.profile, self.assignment]:
            if sub_obj is not None:
                all_keys.extend(f.name for f in fields(sub_obj))

        return all_keys

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]
