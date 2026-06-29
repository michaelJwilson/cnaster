from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class HMRFResult:
    """
    Encapsulates the fitted parameters, state assignments, and likelihoods
    resulting from the HMM + HMRF pipeline.
    """

    # HMM emission parameters
    new_log_mu: (
        np.ndarray
    )  # shape (n_states, n_clones) or similar depending on clone stacking
    new_p_binom: np.ndarray  # shape (n_states, n_clones)
    new_alphas: np.ndarray  # shape (n_states, n_clones)
    new_taus: np.ndarray  # shape (n_states, n_clones)

    # Posterior and predictions (concatenated across clones context)
    log_gamma: (
        np.ndarray
    )  # shape (n_states, n_obs_stack) - The log posterior probability of each state
    pred_cnv: np.ndarray  # shape (n_obs_stack, ) - The MAP copy-number state estimation

    # Spatial clone assignments for spots
    prev_assignment: (
        np.ndarray
    )  # shape (n_spots, ) - Clone assignment prior to the final iteration
    new_assignment: (
        np.ndarray
    )  # shape (n_spots, ) - Final clone assignment after HMRF smoothing

    # Overall fit quality
    total_llf: float  # Total log-likelihood of the HMRF assignment

    # Optional tracking of missing/lost clones during the fitting process
    assignment_before_reindex: Optional[np.ndarray] = None
