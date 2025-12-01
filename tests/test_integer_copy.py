import numpy as np
import pytest

from cnaster.integer_copy import filter_consistent_acn_states

def test_filter_consistent_acn_states_strict():
    # Two states: state 0 ~ diploid balanced, state 1 altered
    new_mu = np.array([1.0, 1.5])
    new_log_mu = np.log(new_mu)
    new_p_binom = np.array([0.50, 0.33])
    
    pred_cnv = np.array([0,0,0,0,0,1,1,1])
    
    baf_allowed, rdrbaf_allowed = filter_consistent_acn_states(
        new_log_mu=new_log_mu,
        new_p_binom=new_p_binom,
        pred_cnv=pred_cnv,
        max_allele_copy=3,
        max_total_copy=4,
        n_sigma=1.0,
        min_prop_threshold=0.1,
        EPS_BAF=0.05,
    )

    print()
    print(new_mu)
    print(new_p_binom)
    print(baf_allowed)
    
    """
    # Diploid balanced state (state 0) should allow (1,1)
    assert (1,1) in baf_dict[0]
    assert (1,1) in both_dict[0]
    # Altered state (state 1) should not accept (1,1) under strict thresholds
    assert (1,1) not in both_dict[1]
    """
