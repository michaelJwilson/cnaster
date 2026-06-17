from dataclasses import dataclass

from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
import numpy as np


@dataclass
class GenomicBlockCounts:
    """
    Container for aggregated spatial transcriptomic counts over genomic segments.

    Dimensions:
    - n_blocks: Number of genomic intervals/blocks.
    - n_spots: Number of spatial spots/barcodes.
    """

    lengths: np.ndarray  # (n_contigs,) Blocks per contig
    X: np.ndarray  # (n_blocks, 2, n_spots) Raw counts
    base_nb_mean: np.ndarray  # (n_blocks, n_spots) Expected normal baseline
    total_bb_RD: np.ndarray  # (n_blocks, n_spots) Total SNP reads (H0 + H1)

    @property
    def umi_counts(self) -> np.ndarray:
        """Total gene expression UMIs per block and spot. Shape: (n_blocks, n_spots)"""
        return self.X[:, 0, :]

    @property
    def allele_a_counts(self) -> np.ndarray:
        """Haplotype A (H0) counts per block and spot. Shape: (n_blocks, n_spots)"""
        return self.X[:, 1, :]

    @property
    def allele_b_counts(self) -> np.ndarray:
        """Haplotype B (H1) counts per block and spot. Shape: (n_blocks, n_spots)"""
        return self.total_bb_RD - self.allele_a_counts

    @property
    def n_blocks(self) -> int:
        return self.X.shape[0]

    @property
    def n_spots(self) -> int:
        return self.X.shape[2]

    def get_baf(self, fill_value: float = 0.5) -> np.ndarray:
        """Safely calculate B-Allele Frequency, handling division by zero."""
        with np.errstate(divide="ignore", invalid="ignore"):
            baf = self.allele_b_counts / self.total_bb_RD
        return np.nan_to_num(baf, nan=fill_value)

    def get_rdr(self, fill_value: float = 1.0) -> np.ndarray:
        """Safely calculate Read-Depth Ratio relative to normal baseline."""
        with np.errstate(divide="ignore", invalid="ignore"):
            rdr = self.umi_counts / self.base_nb_mean
        return np.nan_to_num(rdr, nan=fill_value)

    # TODO
    # merge_pseudobulk_by_index_mix
