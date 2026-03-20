import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
import matplotlib.pyplot as plt
import pytest
import sys
import argparse
from numba import njit

# --- 1. Generative Model & Likelihoods ---

@jax.jit
def betabin_logpdf(k, n, alpha, beta):
    """Log-likelihood for Beta-Binomial emissions."""
    k = jnp.clip(k, 0, n)
    log_const = gammaln(n + 1) - (gammaln(k + 1) + gammaln(n - k + 1))
    log_kernel = (gammaln(k + alpha) - gammaln(alpha) + 
                  gammaln(n - k + beta) - gammaln(beta) - 
                  (gammaln(n + alpha + beta) - gammaln(alpha + beta)))
    return log_const + log_kernel

@jax.jit
def negbin_logpdf(u, mu, phi):
    """Log-likelihood for Negative Binomial emissions with overdispersion phi.
    Variance = mu + phi * mu^2
    Derived parameters: r = 1/phi, p = 1 / (1 + mu * phi).
    """
    mu = jnp.clip(mu, 1e-6, None)
    phi = jnp.clip(phi, 1e-6, None)
    r = 1.0 / phi
    p = 1.0 / (1.0 + mu * phi)
    
    log_const = gammaln(u + r) - gammaln(r) - gammaln(u + 1.0)
    log_kernel = r * jnp.log(p) + u * jnp.log(1.0 - p)
    return log_const + log_kernel

@njit
def numba_wolff_update(labels, external_field, beta_potts, temp_anneal, adj_array):
    """Numba-compiled Wolff Cluster Update."""
    N, C = external_field.shape
    new_labels = labels.copy()
    p_bond = 1.0 - np.exp(-beta_potts / temp_anneal)
    
    for _ in range(5):  
        root = np.random.randint(0, N)
        old_c = new_labels[root]
        new_c = np.random.randint(0, C)
        if old_c == new_c: 
            continue
        
        cluster = [root]
        queue = [root]
        visited = np.zeros(N, dtype=np.bool_)
        visited[root] = True
        
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj_array[u]:
                if v == -1: continue
                if new_labels[v] == old_c and not visited[v]:
                    if np.random.rand() < p_bond:
                        visited[v] = True
                        cluster.append(v)
                        queue.append(v)
        
        ll_diff = 0.0
        for v in cluster:
            ll_diff += external_field[v, new_c] - external_field[v, old_c]
            
        log_ratio = ll_diff / temp_anneal
        if log_ratio >= 0 or np.log(np.random.rand()) < log_ratio:
            for v in cluster:
                new_labels[v] = new_c
                
    return new_labels

# --- 2. Inference Sub-problems ---

class SpatialCNVInference:
    def __init__(self, N_spots, G_segments, K_states, C_clones, adj_array, t_trans=0.9):
        self.N = N_spots
        self.G = G_segments
        self.K = K_states
        self.C = C_clones
        self.adj_array = adj_array
        
        # HMM Transition Prior
        self.trans_log_probs = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                if i == j:
                    self.trans_log_probs[i, j] = np.log(t_trans)
                else:
                    self.trans_log_probs[i, j] = np.log((1.0 - t_trans) / (self.K - 1))
        
    def get_external_field(self, data, profiles, theta):
        """
        Calculates E[n, c]: log-likelihood of spot n belonging to clone c.
        Returns array of shape (N, C) (Vectorized via JAX).
        """
        u, mu_base, k_baf, n_baf = data
        field = np.zeros((self.N, self.C))
        for c in range(self.C):
            prof_c = profiles[c, :]
            mu_multipliers = theta['mu_multiplier'][prof_c] # shape G
            phi_arr = theta['phi'][prof_c]                  # shape G
            alpha_arr = theta['alpha'][prof_c]
            beta_arr = theta['beta'][prof_c]
            
            mu = mu_base * mu_multipliers                           # shape (N, G)
            ll_nb = negbin_logpdf(u, mu, phi_arr)                   # shape (N, G)
            ll_bb = betabin_logpdf(k_baf, n_baf, alpha_arr, beta_arr)
            field[:, c] = np.array(jnp.sum(ll_nb + ll_bb, axis=1))
        return field

    def wolff_update(self, labels, external_field, beta_potts, temp_anneal):
        """Sub-problem 1: Wolff Cluster Update with MH step favoring external field."""
        return numba_wolff_update(labels, external_field, beta_potts, temp_anneal, self.adj_array)

    def stoch_forward_backward(self, labels, data, profiles, theta, temp_anneal):
        """Sub-problem 2: Stochastic update for clone profiles resolving an HMM prior."""
        u, mu_base, k_baf, n_baf = data
        new_profiles = np.copy(profiles)
        
        # Gibbs update for profiles over segments utilizing the Markov blanket
        for c in range(self.C):
            spots_in_c = np.where(labels == c)[0]
            if len(spots_in_c) == 0:
                continue
                
            for g in range(self.G):
                state_log_probs = np.zeros(self.K)
                for k in range(self.K):
                    mu = mu_base[spots_in_c, g] * theta['mu_multiplier'][k]
                    phi_k = theta['phi'][k]
                    ll_nb = float(jnp.sum(negbin_logpdf(u[spots_in_c, g], mu, phi_k)))
                    ll_bb = float(jnp.sum(betabin_logpdf(k_baf[spots_in_c, g], n_baf[spots_in_c, g], theta['alpha'][k], theta['beta'][k])))
                    ll = ll_nb + ll_bb
                    
                    prior_ll = 0.0
                    if g > 0:
                        prior_ll += self.trans_log_probs[new_profiles[c, g-1], k]
                    if g < self.G - 1:
                        prior_ll += self.trans_log_probs[k, new_profiles[c, g+1]]
                        
                    state_log_probs[k] = (ll + prior_ll) / temp_anneal
                
                # Softmax sampling
                state_log_probs -= np.max(state_log_probs)
                probs = np.exp(state_log_probs)
                probs /= probs.sum()
                new_profiles[c, g] = np.random.choice(self.K, p=probs)
                
        return new_profiles

    def empirical_gibbs_theta(self, labels, profiles, data, theta, temp_anneal, fix_nb=False, fix_bb=False, M_candidates=20):
        """
        Sub-problem 3: Fitting Emission Variables theta.
        1. Isolate data D_k mapped to state k.
        2. Propose candidate parameters derived from empirical distributions of subsets.
        3. Gibbs sample the new parameter weighted by total log-likelihood scaled by 1/T.
        """
        u, mu_base, k_baf, n_baf = data
        
        # Copy all existing parameters from theta
        new_theta = {k_key: np.copy(v_val) for k_key, v_val in theta.items()}
        
        # Map every (n, g) to its assigned state k
        assigned_states = profiles[labels, :]
        
        for k in range(self.K):
            # 1. Isolate data D_k
            n_idx, g_idx = np.where(assigned_states == k)
            if len(n_idx) == 0:
                continue
                
            if not fix_nb:
                u_k = u[n_idx, g_idx]
                mu_base_k = mu_base[n_idx, g_idx]
                
                # 2. Propose candidate parameters for NB
                candidates = np.zeros(M_candidates)
                for m in range(M_candidates):
                    subset_size = max(1, len(u_k) // 5)
                    sub_idx = np.random.choice(len(u_k), size=subset_size, replace=True)
                    sum_u = u_k[sub_idx].sum()
                    sum_mu = mu_base_k[sub_idx].sum()
                    candidates[m] = (sum_u + 1e-3) / (sum_mu + 1e-3)
                    
                # 3. Select new parameter by Gibbs sampling over the full D_k
                log_weights = np.zeros(M_candidates)
                for m in range(M_candidates):
                    cand_mu = mu_base_k * candidates[m]
                    ll_m = negbin_logpdf(u_k, cand_mu, theta['phi'][k]).sum()
                    log_weights[m] = ll_m / temp_anneal
                    
                log_weights -= np.max(log_weights)
                probs = np.exp(log_weights)
                probs /= probs.sum()
                selected_idx = np.random.choice(M_candidates, p=probs)
                new_theta['mu_multiplier'][k] = candidates[selected_idx]
                
            if not fix_bb:
                k_k = k_baf[n_idx, g_idx]
                n_k = n_baf[n_idx, g_idx]
                
                sum_k = k_k.sum()
                sum_n = n_k.sum()
                emp_p = (sum_k + 1e-3) / (sum_n + 2e-3)
                
                # Propose overdispersion magnitudes M = alpha + beta
                M_cands = np.array([2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
                log_weights_bb = np.zeros(len(M_cands))
                for m_idx, M_val in enumerate(M_cands):
                    cand_a = emp_p * M_val
                    cand_b = (1.0 - emp_p) * M_val
                    ll_bb = betabin_logpdf(k_k, n_k, cand_a, cand_b).sum()
                    log_weights_bb[m_idx] = ll_bb / temp_anneal
                    
                log_weights_bb -= np.max(log_weights_bb)
                probs_bb = np.exp(log_weights_bb)
                probs_bb /= probs_bb.sum()
                best_M = np.random.choice(M_cands, p=probs_bb)
                
                new_theta['alpha'][k] = emp_p * best_M
                new_theta['beta'][k] = (1.0 - emp_p) * best_M
            
        return new_theta

# --- 3. Simulated Annealing Loop ---

def run_annealing(model, data, steps=100, true_labels=None, true_profiles=None, true_theta=None, fix_labels=False, fix_profiles=False, fix_theta_nb=False, fix_theta_bb=False, beta_potts=1.5):
    """Runs the joint annealing process across all sub-problems, with options to fix specific variables."""
    # Initialize 
    if fix_labels and true_labels is not None:
        labels = np.copy(true_labels)
    else:
        labels = np.random.randint(0, model.C, size=model.N)
        
    if fix_profiles and true_profiles is not None:
        profiles = np.copy(true_profiles)
    else:
        profiles = np.random.randint(0, model.K, size=(model.C, model.G))
        
    # Copy true parameters or initialize
    theta = {
        'mu_multiplier': np.copy(true_theta['mu_multiplier']) if (fix_theta_nb and true_theta) else np.linspace(0.5, 2.0, model.K),
        'phi': np.copy(true_theta['phi']) if (fix_theta_nb and true_theta) else np.ones(model.K) * 0.1,
        'alpha': np.copy(true_theta['alpha']) if (fix_theta_bb and true_theta) else np.ones(model.K) * 2.0,
        'beta': np.copy(true_theta['beta']) if (fix_theta_bb and true_theta) else np.ones(model.K) * 2.0
    }
    
    T_start = 5.0
    T_end = 0.01
    schedule = np.logspace(np.log10(T_start), np.log10(T_end), steps)
    
    for t in range(steps):
        T = schedule[t]
        
        if not fix_labels:
            field = model.get_external_field(data, profiles, theta)
            labels = model.wolff_update(labels, field, beta_potts, T)
        if not fix_profiles:
            profiles = model.stoch_forward_backward(labels, data, profiles, theta, T)
        if not (fix_theta_nb and fix_theta_bb):
            theta = model.empirical_gibbs_theta(labels, profiles, data, theta, T, fix_nb=fix_theta_nb, fix_bb=fix_theta_bb)
            
    return labels, profiles, theta

# --- 4. Shared Utilities & Tests ---

def generate_spatial_mock_data(N_W=25, N_H=25, G=50, K=3, C=3, t_retention=0.90, J_true=1.0):
    """Helper to generate mock spatial CNV data."""
    np.random.seed(42)
    N = N_W * N_H
    coords = np.array([(i % N_W, i // N_W) for i in range(N)])
    adj_array = np.full((N, 4), -1, dtype=np.int32)
    idx_track = np.zeros(N, dtype=int)
    for i in range(N):
        x, y = coords[i]
        if x > 0: adj_array[i, idx_track[i]] = i - 1; idx_track[i] += 1
        if x < N_W - 1: adj_array[i, idx_track[i]] = i + 1; idx_track[i] += 1
        if y > 0: adj_array[i, idx_track[i]] = i - N_W; idx_track[i] += 1
        if y < N_H - 1: adj_array[i, idx_track[i]] = i + N_W; idx_track[i] += 1
        
    model = SpatialCNVInference(N, G, K, C, adj_array, t_trans=t_retention)
    
    true_labels = np.random.randint(0, C, size=N)
    dummy_field = np.zeros((N, C))
    for _ in range(20): 
        true_labels = model.wolff_update(true_labels, dummy_field, J_true, 1.0)
    
    true_profiles = np.zeros((C, G), dtype=int)
    for c in range(C):
        true_profiles[c, 0] = np.random.randint(K)
        for g in range(1, G):
            if np.random.rand() < t_retention:
                true_profiles[c, g] = true_profiles[c, g-1]
            else:
                others = [k for k in range(K) if k != true_profiles[c, g-1]]
                true_profiles[c, g] = np.random.choice(others)
    
    true_theta = {
        'mu_multiplier': np.array([0.5, 1.0, 1.5]),
        'phi': np.array([0.05, 0.1, 0.2]),
        'alpha': np.array([2.0, 10.0, 18.0]),
        'beta': np.array([18.0, 10.0, 2.0])
    }
    
    # T_n: Spot coverage (scaled up by G to keep segment depth comparable after lambda_g normalization)
    eff_cov = np.random.uniform(500.0 * G, 1500.0 * G, size=N) 
    # \lambda_g: Baseline normal expression, normalized to sum to 1
    rel_cov = np.random.uniform(0.5, 2.0, size=G)
    rel_cov /= rel_cov.sum()
    
    mu_base = np.outer(eff_cov, rel_cov)
    
    u_data = np.zeros((N, G))
    n_baf = np.random.poisson(300, size=(N, G))
    k_baf = np.zeros((N, G))
    
    for spot in range(N):
        for g in range(G):
            st = true_profiles[true_labels[spot], g]
            mu = mu_base[spot, g] * true_theta['mu_multiplier'][st]
            phi = true_theta['phi'][st]
            r_nb = 1.0 / phi
            p_nb = 1.0 / (1.0 + mu * phi)
            u_data[spot, g] = np.random.negative_binomial(r_nb, p_nb)
            
            p_baf = np.random.beta(true_theta['alpha'][st], true_theta['beta'][st])
            k_baf[spot, g] = np.random.binomial(n_baf[spot, g], p_baf)
            
    data = (jnp.array(u_data), jnp.array(mu_base), jnp.array(k_baf), jnp.array(n_baf))
    return model, data, true_labels, true_profiles, true_theta, coords, J_true, t_retention

@pytest.mark.parametrize("inference_target", ["labels", "profiles", "theta_nb", "theta_bb", "theta", "all"])
def test_inference_subproblems(inference_target):
    """Test inference of components selectively or jointly, minimizing repetitive code."""
    model, data, true_labels, true_profiles, true_theta, coords, J_true, t_retention = generate_spatial_mock_data()
    
    # Fix variables based on the target of inference
    fix_labels = inference_target not in ("labels", "all")
    fix_profiles = inference_target not in ("profiles", "all")
    fix_theta_nb = inference_target not in ("theta_nb", "theta", "all")
    fix_theta_bb = inference_target not in ("theta_bb", "theta", "all")
    
    inferred_labels, inferred_profiles, inferred_theta = run_annealing(
        model, data, steps=30, 
        true_labels=true_labels, true_profiles=true_profiles, true_theta=true_theta,
        fix_labels=fix_labels, fix_profiles=fix_profiles, fix_theta_nb=fix_theta_nb, fix_theta_bb=fix_theta_bb,
        beta_potts=J_true
    )
    
    u, mu_base, k_baf, n_baf = data
    N, G, K, C = model.N, model.G, model.K, model.C
    k_baf_np = np.array(k_baf)
    n_baf_np = np.array(n_baf)
    u_np = np.array(u)
    mu_base_np = np.array(mu_base)
    
    def calc_ll(lbls, profs, th):
        ll = 0.0
        for n in range(N):
            for g in range(G):
                st = profs[lbls[n], g]
                ll += negbin_logpdf(u[n, g], mu_base[n, g] * th['mu_multiplier'][st], th['phi'][st])
                ll += betabin_logpdf(k_baf[n, g], n_baf[n, g], th['alpha'][st], th['beta'][st])
        return float(ll)

    print(f"\n--- Target: {inference_target.upper()} ---")
    print(f"Log-Likelihood Truth: {calc_ll(true_labels, true_profiles, true_theta):.2f}")
    print(f"Log-Likelihood Inferred: {calc_ll(inferred_labels, inferred_profiles, inferred_theta):.2f}")

    overlap = np.zeros((C, C))
    for c_true in range(C):
        for c_inf in range(C):
            overlap[c_true, c_inf] = np.sum((true_labels == c_true) & (inferred_labels == c_inf))
    
    c_map = {c_inf: np.argmax(overlap[:, c_inf]) for c_inf in range(C)}
    aligned_labels = np.array([c_map[lbl] for lbl in inferred_labels])
    
    true_states = true_profiles[true_labels, :].flatten()
    inf_states = inferred_profiles[inferred_labels, :].flatten()
    
    def print_metrics(true_vals, inf_vals, n_classes, name):
        for cls in range(n_classes):
            tp = np.sum((true_vals == cls) & (inf_vals == cls))
            fn = np.sum((true_vals == cls) & (inf_vals != cls))
            tn = np.sum((true_vals != cls) & (inf_vals != cls))
            fp = np.sum((true_vals != cls) & (inf_vals == cls))
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            print(f"{name} {cls} - Recall: {rec:.2f}, Specificity: {spec:.2f}")

    print(f"\nClone Label Metrics{' (fixed=True)' if fix_labels else ''}:")
    print_metrics(true_labels, aligned_labels, C, "Clone")
    print(f"\nCopy State Metrics{' (fixed=True)' if fix_profiles else ''}:")
    print_metrics(true_states, inf_states, K, "State")

    fixed_params_info = []
    if fix_theta_nb: fixed_params_info.append("nb=Fixed")
    if fix_theta_bb: fixed_params_info.append("bb=Fixed")
    fixed_params_str = f" ({', '.join(fixed_params_info)})" if fixed_params_info else ""
    print(f"\nState Parameters (Truth vs Inferred){fixed_params_str}:")
    
    for k in range(K):
        true_p = true_theta['alpha'][k] / (true_theta['alpha'][k] + true_theta['beta'][k])
        inf_p = inferred_theta['alpha'][k] / (inferred_theta['alpha'][k] + inferred_theta['beta'][k])
        true_tau = true_theta['alpha'][k] + true_theta['beta'][k]
        inf_tau = inferred_theta['alpha'][k] + inferred_theta['beta'][k]
        
        print(f"  State {k}:")
        print(f"    {'mu_multiplier:':<15}\tTrue={true_theta['mu_multiplier'][k]:<6.2f}\tInf={inferred_theta['mu_multiplier'][k]:<6.2f}")
        print()
        print(f"    {'NB phi:':<15}\tTrue={true_theta['phi'][k]:<6.3f}\tInf={inferred_theta['phi'][k]:<6.3f}")
        print(f"    {'NB r (1/phi):':<15}\tTrue={1.0/true_theta['phi'][k]:<6.2f}\tInf={1.0/inferred_theta['phi'][k]:<6.2f}")
        print()
        print(f"    {'BB p (a/a+b):':<15}\tTrue={true_p:<6.3f}\tInf={inf_p:<6.3f}")
        print(f"    {'BB tau (a+b):':<15}\tTrue={true_tau:<6.2f}\tInf={inf_tau:<6.2f}")
        print()

    # Visualizations
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    target_name_map = {"theta_bb": "BB", "theta_nb": "NB", "theta": "NB & BB", "labels": "Labels", "profiles": "Profiles", "all": "All"}
    fig.suptitle(f"Inferred {target_name_map.get(inference_target, inference_target)}", fontsize=18, fontweight='bold')
    
    cmap_discrete = plt.cm.get_cmap('tab10', C)
    
    sc1 = axes[0, 0].scatter(coords[:, 0], coords[:, 1], c=true_labels, cmap=cmap_discrete, vmin=-0.5, vmax=C-0.5, s=100, marker='s')
    axes[0, 0].set_title(f"True Clones (Potts Prior, J={J_true})")
    axes[0, 0].invert_yaxis()
    plt.colorbar(sc1, ax=axes[0, 0], ticks=range(C), label="Clone")

    sc2 = axes[0, 1].scatter(coords[:, 0], coords[:, 1], c=aligned_labels, cmap=cmap_discrete, vmin=-0.5, vmax=C-0.5, s=100, marker='s')
    axes[0, 1].set_title(f"Inferred Clones (Assumed J={J_true})")
    axes[0, 1].invert_yaxis()
    plt.colorbar(sc2, ax=axes[0, 1], ticks=range(C), label="Clone")
    
    segments = np.arange(G)
    axes[1, 0].set_title(f"True BAF (t_retention={t_retention})")
    axes[1, 1].set_title(f"Inferred BAF (t_retention={t_retention})")
    
    spot_baf = k_baf_np / np.maximum(n_baf_np, 1)
    
    for c in range(C):
        spots_true = np.where(true_labels == c)[0]
        if len(spots_true) > 0:
            color = cmap_discrete(c)
            x_vals = np.repeat(segments[np.newaxis, :], len(spots_true), axis=0).flatten()
            y_vals = spot_baf[spots_true, :].flatten()
            axes[1, 0].scatter(x_vals + np.random.uniform(-0.2, 0.2, size=len(x_vals)), y_vals, color=color, alpha=0.5, s=10, marker='.')
            true_p = [true_theta['alpha'][true_profiles[c, g]] / (true_theta['alpha'][true_profiles[c, g]] + true_theta['beta'][true_profiles[c, g]]) for g in segments]
            axes[1, 0].plot(segments, true_p, linestyle='-', linewidth=2, color=color, label=f"Clone {c}")

    for c_inf in range(C):
        spots_inf = np.where(inferred_labels == c_inf)[0]
        if len(spots_inf) > 0:
            c_mapped = c_map[c_inf]
            color = cmap_discrete(c_mapped)
            x_vals = np.repeat(segments[np.newaxis, :], len(spots_inf), axis=0).flatten()
            y_vals = spot_baf[spots_inf, :].flatten()
            axes[1, 1].scatter(x_vals + np.random.uniform(-0.2, 0.2, size=len(x_vals)), y_vals, color=color, alpha=0.5, s=10, marker='.')
            
            inf_profile = inferred_profiles[c_inf]
            inf_p = []
            inf_p_std = []
            for g in segments:
                a = inferred_theta['alpha'][inf_profile[g]]
                b = inferred_theta['beta'][inf_profile[g]]
                p = a / (a + b)
                avg_n = np.mean(n_baf_np[spots_inf, g])
                # Beta-Binomial variance for k/n
                var_bb = (p * (1 - p) / avg_n) * (1 + (avg_n - 1) / (a + b + 1))
                
                inf_p.append(p)
                inf_p_std.append(np.sqrt(var_bb))
                
            inf_p = np.array(inf_p)
            inf_p_std = np.array(inf_p_std)
            
            axes[1, 1].plot(segments, inf_p, linestyle='-', linewidth=2, color=color, label=f"Clone {c_mapped}")
            axes[1, 1].fill_between(segments, inf_p - inf_p_std, inf_p + inf_p_std, color=color, alpha=0.3)
            
    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_xlabel("g")
        ax.set_ylabel(r"BAF ($k/n$)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize="small", loc="upper right")

    # --- Panel 3 (Bottom): True vs Data RDR (u/mu_base) ---
    axes[2, 0].set_title("True RDR")
    axes[2, 1].set_title("Inferred RDR")
    
    spot_rdr = u_np / np.maximum(mu_base_np, 1e-3)
    
    for c in range(C):
        spots_true = np.where(true_labels == c)[0]
        if len(spots_true) > 0:
            color = cmap_discrete(c)
            x_vals = np.repeat(segments[np.newaxis, :], len(spots_true), axis=0).flatten()
            y_vals = spot_rdr[spots_true, :].flatten()
            axes[2, 0].scatter(x_vals + np.random.uniform(-0.2, 0.2, size=len(x_vals)), y_vals, color=color, alpha=0.5, s=10, marker='.')
            true_rdr_p = [true_theta['mu_multiplier'][true_profiles[c, g]] for g in segments]
            axes[2, 0].plot(segments, true_rdr_p, linestyle='-', linewidth=2, color=color, label=f"Clone {c}")

    for c_inf in range(C):
        spots_inf = np.where(inferred_labels == c_inf)[0]
        if len(spots_inf) > 0:
            c_mapped = c_map[c_inf]
            color = cmap_discrete(c_mapped)
            x_vals = np.repeat(segments[np.newaxis, :], len(spots_inf), axis=0).flatten()
            y_vals = spot_rdr[spots_inf, :].flatten()
            axes[2, 1].scatter(x_vals + np.random.uniform(-0.2, 0.2, size=len(x_vals)), y_vals, color=color, alpha=0.5, s=10, marker='.')
            
            inf_profile = inferred_profiles[c_inf]
            inf_rdr_p = []
            inf_rdr_std = []
            for g in segments:
                m = inferred_theta['mu_multiplier'][inf_profile[g]]
                phi = inferred_theta['phi'][inf_profile[g]]
                mb = np.mean(mu_base_np[spots_inf, g])
                # Negative Binomial variance for u/mb = (mb*m + phi*(mb*m)^2) / mb^2 = m/mb + phi*m^2
                var_rdr = (m / mb) + phi * (m**2)
                
                inf_rdr_p.append(m)
                inf_rdr_std.append(np.sqrt(var_rdr))
                
            inf_rdr_p = np.array(inf_rdr_p)
            inf_rdr_std = np.array(inf_rdr_std)
            
            axes[2, 1].plot(segments, inf_rdr_p, linestyle='-', linewidth=2, color=color, label=f"Clone {c_mapped}")
            axes[2, 1].fill_between(segments, inf_rdr_p - inf_rdr_std, inf_rdr_p + inf_rdr_std, color=color, alpha=0.3)
            
    for ax in [axes[2, 0], axes[2, 1]]:
        ax.set_xlabel("g")
        ax.set_ylabel(r"RDR ($u / T_n \lambda_g$)")
        ax.legend(fontsize="small", loc="upper right")
    
    plt.tight_layout()
    # Adjust layout to make room for the suptitle
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"test_inference_inferring_{inference_target}_output.png")
    plt.close()

    # Simple verification checks ensures frozen parameters didn't drift
    if fix_labels:
        assert np.array_equal(inferred_labels, true_labels)
    if fix_profiles:
        assert np.array_equal(inferred_profiles, true_profiles)
    if fix_theta_nb:
        assert np.array_equal(inferred_theta['mu_multiplier'], true_theta['mu_multiplier'])
    if fix_theta_bb:
        assert np.array_equal(inferred_theta['alpha'], true_theta['alpha'])
        assert np.array_equal(inferred_theta['beta'], true_theta['beta'])

    assert inferred_labels.shape == (N,)
    assert set(np.unique(inferred_labels)).issubset(set(range(C)))
    assert inferred_profiles.shape == (C, G)
    assert len(inferred_theta['mu_multiplier']) == K

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spatial CNV Inference Test")
    parser.add_argument("--target", type=str, default=None, 
                        help="Inference target: labels, profiles, theta_nb, theta_bb, theta, all. If not provided, runs all.")
    args = parser.parse_args()

    targets = ["labels", "profiles", "theta_nb", "theta_bb", "theta", "all"]
    if args.target:
        if args.target in targets:
            test_inference_subproblems(args.target)
        else:
            print(f"Invalid target. Choose from {targets}")
    else:
        for t in targets:
            test_inference_subproblems(t)
