from dataclasses import dataclass

from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
import numpy as np


@dataclass
class SpatioGenomicCounts:
    """
    Data container for spatial transcriptomic counts aggregated over genomic segments.

    Dimensions:

    - n_segments: number of genomic intervals/segments.
    - n_spots: number of spatial spots/barcodes.
    """
    lengths: np.ndarray  # (n_contigs,) num. segments per contig
    X: np.ndarray  # (n_segments, 2, n_spots) observed counts, 0: genes, 1: snps
    base_nb_mean: (
        np.ndarray
    )  # (n_segments, n_spots) expected baseline expression for normal cells
    total_bb_RD: (
        np.ndarray
    )  # (n_segments, n_spots) total (both haplotypes) snp-covering reads in segment

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
