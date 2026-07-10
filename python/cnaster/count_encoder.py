import numpy as np
import scipy

from cnaster.config import get_global_config, start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


class CountEncoder:
    def __init__(self, obs_count, total_count, common_zero_depth=False):
        self.obs_count = obs_count
        self.total_count = total_count

        self.n_obs = obs_count.shape[0]
        self.n_spots = obs_count.shape[1]

        # NB [unique counts, ...] and [mapping_matrices, ...] by spot.
        self.unique_counts, self.mapping_matrices = self.construct_unique_encoding(
            obs_count, total_count, common_zero_depth=common_zero_depth
        )

    def encode_array(self, array, spot):
        # mapper is (nobs, nunique),
        # array is  [..., nobs] or [nobs, ...]
        # result is [..., nunique] or [nunique, ...]
        mapper = self.mapping_matrices[spot]

        # NB sparse transpose is cheap
        if array.shape[-1] == mapper.shape[0]:
            return array @ mapper
        elif array.shape[0] == mapper.shape[0]:
            return mapper.T @ array
        else:
            raise ValueError(
                f"Array shape {array.shape} not compatible with mapper shape {mapper.shape}."
            )

    def decode_array(self, array, spot):
        # mapper is (nobs, nunique),
        # array is encoded as [..., nunique] or [nunique, ...]
        # result is [..., nobs] or [nobs, ...]
        mapper = self.mapping_matrices[spot]

        if array.shape[-1] == mapper.shape[-1]:
            return array @ mapper.T
        elif array.shape[0] == mapper.shape[-1]:
            return mapper @ array
        else:
            raise ValueError(
                f"Array shape {array.shape} not compatible with mapper shape {mapper.shape}."
            )

    def get_unique_obs(self, spot):
        return self.unique_counts[spot][:, 0]

    def get_unique_total(self, spot):
        return self.unique_counts[spot][:, 1]

    @property
    def compression_rate(self):
        total_original_elements = self.n_obs * self.n_spots

        if total_original_elements == 0:
            return 0.0

        total_compressed_elements = sum(u.shape[0] for u in self.unique_counts)

        return 1.0 - total_compressed_elements / total_original_elements

    @staticmethod
    def construct_unique_encoding(obs_count, total_count, common_zero_depth=True):
        decimals = get_global_config().hmm.compression_decimals
        unique_values, mapping_matrices = [], []

        n_obs = obs_count.shape[0]
        n_spots = obs_count.shape[1]

        for s in range(n_spots):
            o_col = obs_count[:, s].copy()
            t_col = total_count[:, s].copy()

            # Collapse all entries with 0 total counts into a single (0, 0) state
            if common_zero_depth:
                zero_mask = t_col == 0
                o_col[zero_mask] = 0
                t_col[zero_mask] = 0

            counts = np.vstack([o_col, t_col]).T

            # Fixed np.issubdtype check
            if not np.issubdtype(total_count.dtype, np.integer):
                counts = counts.round(decimals=decimals)

            # NB return_inverse=True yields the exact column indices for mapping,
            # avoiding the slow dictionary lookup loop and rounding inconsistency.
            pairs, inv_indices = np.unique(counts, axis=0, return_inverse=True)
            unique_values.append(pairs)

            # NB construct mapping matrix with shape (n_obs, n_unique_pairs);
            #    one-hot of obs. to compressed.
            mat_row = np.arange(n_obs)
            mat_col = inv_indices

            csr_matrix = scipy.sparse.csr_matrix(
                (np.ones(len(mat_row)), (mat_row, mat_col)),
                shape=(n_obs, pairs.shape[0]),
            )

            # Example usage:
            #   e.g.  convert posteriors from observation space to the compressed space
            # .        tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
            mapping_matrices.append(csr_matrix)

        return unique_values, mapping_matrices
