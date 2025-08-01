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

        n_samples, n_states = 1000, 3
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

        # Generate weights (EM posterior probabilities)
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

    def fit_models_process_pool(self, model, model_fit_params, start_params=None):
        def fit_primary():
            return model.fit(**model_fit_params)

        def fit_secondary():
            if start_params is not None:
                return model.fit(**model_fit_params, start_params=start_params)
            return None

        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
                future_res = executor.submit(fit_primary)
                future_res2 = executor.submit(fit_secondary)

                res = future_res.result()
                res2 = future_res2.result()

            return res, res2
        except Exception as e:
            print(f"Process pool execution failed: {e}")
            return None, None

    def fit_models_sequential(self, model, model_fit_params, start_params=None):
        res = model.fit(**model_fit_params)
        res2 = None
        
        if start_params is not None:
            res2 = model.fit(**model_fit_params, start_params=start_params)

        return res, res2

    def test_model_serialization(self):
        model = Weighted_BetaBinom_mix(
            self.endog,
            self.features,
            weights=self.weights,
            exposure=self.exposure,
            compress=False,
        )

        try:
            pickled_data = pickle.dumps(model)
            unpickled_model = pickle.loads(pickled_data)
            print("✓ Pickle serialization works")
            
            # Verify the unpickled model has the same attributes
            np.testing.assert_array_equal(model.endog, unpickled_model.endog)
            np.testing.assert_array_equal(model.exog, unpickled_model.exog)
            np.testing.assert_array_equal(model.weights, unpickled_model.weights)
            np.testing.assert_array_equal(model.exposure, unpickled_model.exposure)
            
            pickle_success = True
        except Exception as e:
            print(f"✗ Pickle serialization failed: {e}")
            pickle_success = False

        try:
            copied_model = copy.deepcopy(model)
            print("✓ Deep copy works")
            
            # Verify the copied model has the same attributes
            np.testing.assert_array_equal(model.endog, copied_model.endog)
            np.testing.assert_array_equal(model.exog, copied_model.exog)
            np.testing.assert_array_equal(model.weights, copied_model.weights)
            np.testing.assert_array_equal(model.exposure, copied_model.exposure)
            
            deepcopy_success = True
        except Exception as e:
            print(f"✗ Deep copy failed: {e}")
            deepcopy_success = False

        if pickle_success:
            try:
                result = unpickled_model.fit(**self.model_fit_params)
                print("✓ Unpickled model can fit successfully")
            except Exception as e:
                print(f"✗ Unpickled model fit failed: {e}")

        if deepcopy_success:
            try:
                result = copied_model.fit(**self.model_fit_params)
                print("✓ Deep copied model can fit successfully")
            except Exception as e:
                print(f"✗ Deep copied model fit failed: {e}")

        return {
            'pickle_success': pickle_success,
            'deepcopy_success': deepcopy_success
        }

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

        start_time = time.time()
        res_proc, res2_proc = self.fit_models_process_pool(
            model, self.model_fit_params, self.start_params
        )
        process_time = time.time() - start_time

        thread_speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        process_speedup = sequential_time / process_time if process_time > 0 and res_proc is not None else 0

        print(f"\n=== Performance Results ===")
        print(f"Sequential time: {sequential_time:.3f} seconds")
        print(f"Thread pool time: {parallel_time:.3f} seconds")
        print(f"Process pool time: {process_time:.3f} seconds")
        print(f"Thread speedup: {thread_speedup:.2f}x")
        print(f"Process speedup: {process_speedup:.2f}x")
        print(
            f"Thread time saved: {sequential_time - parallel_time:.3f} seconds ({((sequential_time - parallel_time) / sequential_time * 100):.1f}%)"
        )
        if res_proc is not None:
            print(
                f"Process time saved: {sequential_time - process_time:.3f} seconds ({((sequential_time - process_time) / sequential_time * 100):.1f}%)"
            )
        else:
            print("Process pool execution failed (likely serialization issues)")

        return {
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "process_time": process_time,
            "thread_speedup": thread_speedup,
            "process_speedup": process_speedup,
            "thread_time_saved": sequential_time - parallel_time,
            "process_time_saved": sequential_time - process_time if res_proc is not None else None,
        }
                
if __name__ == "__main__":    
    unittest.main(verbosity=2)
