import numpy as np
import scipy
from cnaster.config import get_global_config
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

class CountEncoder:
    def __init__(self, obs_count, total_count):
        self.obs_count = obs_count
        self.total_count = total_count

        self.n_obs = obs_count.shape[0]
        self.n_spots = obs_count.shape[1]

        # NB [unique counts, ...] and [mapping_matrices, ...] by spot.
        self.unique_counts, self.mapping_matrices = self.construct_unique_encoding(
            obs_count, total_count
        )

    def encode_vector(self, vector, spot):
        assert len(vector) == self.n_obs

        mapper = self.mapping_matrices[spot]

        return (mapper.T @ vector).flatten()

    def decode_vector(self, vector, spot):
        assert len(vector) == self.mapping_matrices[spot].shape[1]

        mapper = self.mapping_matrices[spot]

        return (mapper @ vector).flatten()

    @staticmethod
    def construct_unique_encoding(obs_count, total_count):
        decimals = get_global_config().hmm.compression_decimals
        unique_values, mapping_matrices = [], []

        n_obs = obs_count.shape[0]
        n_spots = obs_count.shape[1]

        for s in range(n_spots):
            counts = np.vstack([obs_count[:, s], total_count[:, s]]).T

            # TODO BUG fails for numpy cases; not np.issubdtype(total_count.dtype, np.integer)
            if total_count.dtype != int:
                counts = counts.round(decimals=decimals)

            # NB unique (rounded) pairs of (obs_count, total_count) for spot s.
            pairs, _ = np.unique(counts, axis=0, return_counts=True)
            unique_values.append(pairs)

            # NB mapper of unique pairs to idx.
            pair_index = {(pairs[i, 0], pairs[i, 1]): i for i in range(pairs.shape[0])}

            # NB construct mapping matrix with shape (n_obs, n_unique_pairs);
            #    one-hot of obs. to compressed.
            mat_row = np.arange(n_obs)

            # NB each observation gets the index of its corresponding unique pair.
            mat_col = np.zeros(n_obs, dtype=int)

            for i in range(n_obs):
                if total_count.dtype == int:
                    tmpidx = pair_index[(obs_count[i, s], total_count[i, s])]
                else:
                    # TODO inconsistent with rounding of counts above, i.e. no obs rounding.
                    tmpidx = pair_index[
                        (obs_count[i, s], total_count[i, s].round(decimals=decimals))
                    ]
                mat_col[i] = tmpidx

            # NB num. columns set by max(mat_col).
            csr_matrix = scipy.sparse.csr_matrix(
                (np.ones(len(mat_row)), (mat_row, mat_col))
            )

            # Example usage:
            #   e.g.  convert posteriors from observation space to the compressed space
            # .        tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
            mapping_matrices.append(csr_matrix)

        # NB unique_values is a list of length n_spots, each element is an array of shape (n_unique_pairs, 2) with columns of rounded (obs_count, total_count).
        #    mapping_matrices is a list of length n_spots, each element is a sparse matrix of shape (n_obs, n_unique_pairs) mapping obs. to compressed space.
        return unique_values, mapping_matrices
