import numpy as np
import pytest
from scipy.sparse import csr_matrix
from cnaster.hmrf import pool_hmrf_data
from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.hmrf import aggr_hmrfmix_reassignment_concatenate
from cnaster.deprecated.hmrf import (
    aggr_hmrfmix_reassignment_concatenate as deprecated_aggr_hmrfmix_reassignment_concatenate,
)


class TestHmrf:
    def setup_method(self):
        np.random.seed(42)

        self.n_obs = 1_000
        self.n_spots = 100
        self.n_states = 5
        self.n_clones = 4

        # TODO explain dimensions
        single_X_shape = (self.n_obs, 2, self.n_spots)
        self.single_X = np.random.randint(0, 100, single_X_shape)
        self.single_base_nb_mean = np.random.uniform(1, 10, (self.n_obs, self.n_spots))
        self.single_total_bb_RD = np.random.randint(5, 20, (self.n_obs, self.n_spots))

        # NB create adjacency matrix (each spot connected to 4 neighbors)
        adjacency_data = np.zeros((self.n_spots, self.n_spots))

        for i in range(self.n_spots):
            neighbors = np.random.choice(
                np.delete(np.arange(self.n_spots), i),
                size=4,
                replace=False,
            )
            weights = np.random.uniform(0.5, 1.0, size=len(neighbors))
            adjacency_data[i, neighbors] = weights

        # NB symmetric
        adjacency_data = adjacency_data + adjacency_data.T

        # NB include self-connections
        np.fill_diagonal(adjacency_data, 1.0)

        self.smooth_mat = csr_matrix(adjacency_data)

        # Tumor proportion data
        self.single_tumor_prop = np.random.uniform(0.3, 0.9, self.n_spots)
        self.single_tumor_prop[2] = np.nan

        # NB dummy HMM parameters for mixture model
        self.res_new_log_mu = np.random.normal(1, 10, (self.n_states, self.n_clones))
        self.pred = np.random.randint(0, self.n_states, self.n_obs * self.n_clones)
        self.lambd = np.random.uniform(0.1, 0.3, self.n_obs)
        self.lambd = self.lambd / np.sum(self.lambd)

        self.res = {
            "new_p_binom": np.random.uniform(0.1, 0.9, (self.n_states, self.n_clones)),
            "new_log_mu": np.random.normal(0, 1, (self.n_states, self.n_clones)),
            "new_alphas": np.random.uniform(1, 10, (self.n_states, self.n_clones)),
            "new_taus": np.random.uniform(0.1, 1, (self.n_states, self.n_clones))
        }
        self.adjacency_mat = csr_matrix(adjacency_data)
        self.prev_assignment = np.random.randint(0, self.n_clones, self.n_spots)
        self.sample_ids = np.arange(self.n_spots)
        self.spatial_weight = 0.5

    def test_numba_compilation(self):
        try:
            result = pool_hmrf_data(
                self.single_X,
                self.single_base_nb_mean,
                self.single_total_bb_RD,
                self.smooth_mat.indices,
                self.smooth_mat.indptr,
                use_mixture=False,
            )
            assert result is not None
        except Exception as e:
            pytest.fail(f"Numba compilation failed: {e}")

    def test_equivalence(self):
        exp = deprecated_aggr_hmrfmix_reassignment_concatenate(
            self.single_X,
            self.single_base_nb_mean,
            self.single_total_bb_RD,
            self.res,
            self.pred,
            self.smooth_mat,
            self.adjacency_mat,
            self.prev_assignment,
            self.sample_ids,
            self.spatial_weight,
            log_persample_weights=None,
            single_tumor_prop=None,
            hmmclass=hmm_nophasing,
            return_posterior=False,
        )

        res = aggr_hmrfmix_reassignment_concatenate(
            self.single_X,
            self.single_base_nb_mean,
            self.single_total_bb_RD,
            self.res,
            self.pred,
            self.smooth_mat,
            self.adjacency_mat,
            self.prev_assignment,
            self.sample_ids,
            self.spatial_weight,
            hmmclass=hmm_nophasing,
        )

        # Assert equivalence of results
        np.testing.assert_array_almost_equal(exp, res, decimal=10)


if __name__ == "__main__":
    test_suite = TestHmrf()
    test_suite.setup_method()

    print("Running tests for pool_hmrf_data...")

    try:
        test_suite.test_numba_compilation()
        print("✓ Numba compilation test passed")

        test_suite.test_equivalence()
        print("✓ Equivalence test passed")

        print("\nAll tests passed! 🎉")

    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
