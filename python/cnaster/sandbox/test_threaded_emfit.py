import unittest
import time
import pickle
import copy
import numpy as np
import concurrent.futures
from cnaster.config import YAMLConfig, set_global_config
from cnaster.hmm_emission import Weighted_BetaBinom_mix
from cnaster.hmm_update import get_em_solver_params

# TODO HACK
config = YAMLConfig.from_file("config.yaml")
set_global_config(config)

class TestParallelFitting(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

        n_samples, n_states = 5_000, 5
        state_assignments = np.random.choice(n_states, n_samples)
        
        self.features = np.zeros((n_samples, n_states))
        self.features[np.arange(n_samples), state_assignments] = 1

        self.exposure = 10 + np.random.poisson(50, n_samples)

        true_probs = np.array([0.2, 0.5, 0.8])
        true_tau = 10.0

        self.endog = np.zeros(n_samples)
        for i in range(n_samples):
            state = np.argmax(self.features[i])
            p = true_probs[state]
            a = p * true_tau
            b = (1 - p) * true_tau

            prob = np.random.beta(a, b)
            
            self.endog[i] = np.random.binomial(self.exposure[i], prob)

        self.weights = np.ones(n_samples)

        assert len(self.endog) == len(self.exposure)
        assert len(self.exposure) == len(self.features)
        assert len(self.weights) == len(self.features)
        
        self.model_fit_params = {
            "maxiter": 100,
            "maxfun": 500,
            "legacy": False,
        }
        
        self.start_params = np.array([0.3, 0.6, 0.7, 8.0])  # [p1, p2, p3, tau]

    def fit_models_parallel(self, model, model_fit_params, start_params=None):
        def fit_primary():
            return model.fit(**model_fit_params)

        def fit_secondary():
            if start_params is not None:
                return model.fit(**model_fit_params, start_params=start_params)
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_res = executor.submit(fit_primary)
            future_res2 = executor.submit(fit_secondary)

            res = future_res.result()
            res2 = future_res2.result()

        return res, res2

    def fit_models_sequential(self, model, model_fit_params, start_params=None):
        res = model.fit(**model_fit_params)
        res2 = None
        
        if start_params is not None:
            res2 = model.fit(**model_fit_params, start_params=start_params)

        return res, res2

    def test_parallel_vs_sequential_performance(self):
        model = Weighted_BetaBinom_mix(
            self.endog,
            self.features,
            weights=self.weights,
            exposure=self.exposure,
            compress=False,
        )

        start_time = time.time()
        res_seq, res2_seq = self.fit_models_sequential(
            model, self.model_fit_params, self.start_params
        )
        sequential_time = time.time() - start_time

        start_time = time.time()
        res_par, res2_par = self.fit_models_parallel(
            model, self.model_fit_params, self.start_params
        )
        parallel_time = time.time() - start_time

        thread_speedup = sequential_time / parallel_time if parallel_time > 0 else 0

        # Assert that results match between sequential and parallel execution
        np.testing.assert_allclose(res_seq.params, res_par.params, rtol=1e-10, atol=1e-12,
                                   err_msg="Primary fit parameters don't match between sequential and parallel")
        
        np.testing.assert_allclose(res_seq.llf, res_par.llf, rtol=1e-10, atol=1e-12,
                                   err_msg="Primary fit log-likelihood doesn't match between sequential and parallel")
        
        if res2_seq is not None and res2_par is not None:
            np.testing.assert_allclose(res2_seq.params, res2_par.params, rtol=1e-10, atol=1e-12,
                                       err_msg="Secondary fit parameters don't match between sequential and parallel")
            
            np.testing.assert_allclose(res2_seq.llf, res2_par.llf, rtol=1e-10, atol=1e-12,
                                       err_msg="Secondary fit log-likelihood doesn't match between sequential and parallel")

        print(f"\n=== Performance Results ===")
        print(f"Sequential time: {sequential_time:.3f} seconds")
        print(f"Thread pool time: {parallel_time:.3f} seconds")
        print(f"Thread speedup: {thread_speedup:.2f}x")
        print(
            f"Thread time saved: {sequential_time - parallel_time:.3f} seconds ({((sequential_time - parallel_time) / sequential_time * 100):.1f}%)"
        )
        print("✓ All results match between sequential and parallel execution")

        return {
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "thread_speedup": thread_speedup,
            "thread_time_saved": sequential_time - parallel_time,
        }
                
if __name__ == "__main__":    
    unittest.main(verbosity=2)
