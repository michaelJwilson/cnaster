import copy
import numpy as np
from typing import Optional, Any
from dataclasses import dataclass, fields, is_dataclass
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

class LockableMixin:
    _locked: bool = False

    def lock(self) -> None:
        object.__setattr__(self, '_locked', True)
        
        if is_dataclass(self):
            for f in fields(self):
                val = getattr(self, f.name)
                if isinstance(val, LockableMixin):
                    val.lock()

    def unlock(self) -> None:
        object.__setattr__(self, '_locked', False)
        
        if is_dataclass(self):
            for f in fields(self):
                val = getattr(self, f.name)
                if isinstance(val, LockableMixin):
                    val.unlock()

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False) and name != "_locked":
            raise RuntimeError(f"Instance is locked. Cannot modify attribute '{name}'.")
        super().__setattr__(name, value)

@dataclass
class HMMParams(LockableMixin):
    new_log_mu: np.ndarray
    new_alphas: np.ndarray
    new_p_binom: np.ndarray
    new_taus: np.ndarray
    new_log_startprob: np.ndarray
    new_log_transmat: np.ndarray


@dataclass
class HMMProfile(LockableMixin):
    log_gamma: np.ndarray
    pred_cnv: np.ndarray


@dataclass
class CloneAssignment(LockableMixin):
    assignment_before_reindex: Optional[np.ndarray] = None
    prev_assignment: Optional[np.ndarray] = None
    new_assignment: Optional[np.ndarray] = None
    total_llf: float = np.nan

    @property
    def unique_clone_labels(self):
        if self.assignment is not None and self.assignment.new_assignment is not None:
            return np.unique(self.assignment.new_assignment)
        return None

    @property
    def num_clones(self):
        unique_labels = self.unique_clone_labels
        return len(unique_labels) if unique_labels is not None else None

@dataclass
class CnaHMRFResult(LockableMixin):
    params: HMMParams
    profile: HMMProfile
    llf: float
    n_states: int
    assignment: CloneAssignment

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        # NB valid clone assignment must be 0...N labels
        if self.assignment is not None and getattr(self.assignment, "new_assignment") is not None:
            unique_labels = np.unique(self.assignment.new_assignment)

            if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
                raise ValueError(
                    f"Invalid clone assignment: {unique_labels}. "
                    "Clone labels must be consecutive integers starting from 0."
                )

        # NB log the shape of new_log_mu, log_gamma and new_assignment
        logger.info(
            f"Validated CnaHMRFResult with new_log_mu.shape={self.params.new_log_mu.shape},"
            f"log_gamma.shape={self.profile.log_gamma.shape}, "
            f"pred_cnv.shape={self.profile.pred_cnv.shape}, "
            f"new_assignment.shape={getattr(self.assignment, 'new_assignment', None).shape if getattr(self.assignment, 'new_assignment', None) is not None else None}, "
            f"num_clones={getattr(self.assignment, 'num_clones', 0)}."
        )

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

        self.validate()

        logger.debug(f"Successfully set '{key}' in CnaHMRFResult with new value: {value}.")

    def __str__(self) -> str:
        lines = ["CnaHMRFResult:"]
        
        def format_val(val: Any, is_param: bool, indent: str) -> str:
            if isinstance(val, np.ndarray):
                if is_param:
                    arr_str = np.array2string(val, threshold=np.inf, separator=', ')
                    indented_arr = indent + arr_str.replace('\n', '\n' + indent)
                    return f"\n{indented_arr}"
                else:
                    arr_str = np.array2string(val, threshold=10, edgeitems=2, separator=', ')
                    indented_arr = indent + arr_str.replace('\n', '\n' + indent)
                    return f"<ndarray shape={val.shape} dtype={val.dtype}>\n{indented_arr}"
                    
            elif isinstance(val, float):
                return f"{val:.6f}" if not np.isnan(val) else "nan"
            
            return str(val)

        for f in fields(self):
            if f.name not in {"params", "profile", "assignment"}:
                val = getattr(self, f.name)
                lines.append(f"  {f.name}: {format_val(val, False, '    ')}")

        for nested_name in ["params", "profile", "assignment"]:
            nested_obj = getattr(self, nested_name)
            
            if nested_obj is None:
                lines.append(f"  {nested_name}: None")
                continue
                
            lines.append(f"  {nested_name} ({nested_obj.__class__.__name__}):")
            
            is_param = (nested_name == "params")
            
            for f in fields(nested_obj):
                val = getattr(nested_obj, f.name)
                val_str = format_val(val, is_param, "      ")
                
                if val_str.startswith("\n"):
                    lines.append(f"    {f.name}:{val_str}")
                else:
                    lines.append(f"    {f.name}: {val_str}")

        return "\n".join(lines)

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