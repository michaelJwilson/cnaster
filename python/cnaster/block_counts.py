import copy
from dataclasses import dataclass, fields, is_dataclass
import numpy as np
from typing import Any

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
                elif isinstance(val, np.ndarray):
                    val.flags.writeable = False

    def unlock(self) -> None:
        object.__setattr__(self, '_locked', False)
        
        if is_dataclass(self):
            for f in fields(self):
                val = getattr(self, f.name)
                if isinstance(val, LockableMixin):
                    val.unlock()
                elif isinstance(val, np.ndarray):
                    val.flags.writeable = True

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False) and name != "_locked":
            raise RuntimeError(f"Instance is locked. Cannot modify attribute '{name}'.")
        super().__setattr__(name, value)


@dataclass
class SpatioGenomicCounts(LockableMixin):
    lengths: np.ndarray       # (n_contigs,) num. segments per contig
    X: np.ndarray             # (n_segments, 2, n_spots) observed counts, 0: genes, 1: snps
    base_nb_mean: np.ndarray  # (n_segments, n_spots) expected baseline expression
    total_bb_RD: np.ndarray   # (n_segments, n_spots) total snp-covering reads

    @property
    def n_segments(self) -> int:
        return self.X.shape[0]

    @property
    def n_spots(self) -> int:
        return self.X.shape[2]

    @property
    def transcript_counts(self) -> np.ndarray:
        """Total gene expression UMIs per segment and spot. Shape: (n_segments, n_spots)"""
        return self.X[:, 0, :]

    @property
    def hap_counts(self) -> np.ndarray:
        """Haplotype A (H0) counts per segment and spot. Shape: (n_segments, n_spots)"""
        return self.X[:, 1, :]

    @property
    def alt_hap_counts(self) -> np.ndarray:
        """Haplotype B (H1) counts per segment and spot. Shape: (n_segments, n_spots)"""
        return self.total_bb_RD - self.hap_counts

    def baf(self, fill_value: float = np.nan) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            baf = self.hap_counts / self.total_bb_RD
        return np.nan_to_num(baf, nan=fill_value)

    def rdr(self, fill_value: float = np.nan) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            rdr = self.transcript_counts / self.base_nb_mean
        return np.nan_to_num(rdr, nan=fill_value)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        n_seg = self.n_segments
        n_spt = self.n_spots

        if np.sum(self.lengths) != n_seg:
            raise ValueError(f"Lengths array sum ({np.sum(self.lengths)}) does not match X n_segments ({n_seg}).")

        if self.base_nb_mean.shape != (n_seg, n_spt):
            raise ValueError(f"base_nb_mean shape {self.base_nb_mean.shape} mismatch. Expected: ({n_seg}, {n_spt})")

        if self.total_bb_RD.shape != (n_seg, n_spt):
            raise ValueError(f"total_bb_RD shape {self.total_bb_RD.shape} mismatch. Expected: ({n_seg}, {n_spt})")

        logger.debug(
            f"Validated SpatioGenomicCounts with n_segments={n_seg}, n_spots={n_spt}, "
            f"n_contigs={len(self.lengths)}."
        )

    def __str__(self) -> str:
        lines = ["SpatioGenomicCounts:"]
        
        def format_val(val: Any, indent: str) -> str:
            if isinstance(val, np.ndarray):
                arr_str = np.array2string(val, threshold=10, edgeitems=2, separator=', ')
                indented_arr = indent + arr_str.replace('\n', '\n' + indent)
                return f"<ndarray shape={val.shape} dtype={val.dtype}>\n{indented_arr}"
            return str(val)

        for f in fields(self):
            val = getattr(self, f.name)
            lines.append(f"  {f.name}: {format_val(val, '    ')}")

        return "\n".join(lines)

    # NB mirror dictionary behavior for the top-level attributes
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in SpatioGenomicCounts.")

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
            self.validate()
        else:
            raise KeyError(f"Cannot set unknown key '{key}'")

    def copy(self, deep: bool = False) -> "SpatioGenomicCounts":
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def keys(self):
        return [f.name for f in fields(self)]

    def values(self):
        return [getattr(self, k) for k in self.keys()]

    def items(self):
        return [(k, getattr(self, k)) for k in self.keys()]