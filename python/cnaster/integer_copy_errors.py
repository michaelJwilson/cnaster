import logging
import numpy as np

logger = logging.getLogger(__name__)


def filter_consistent_acn_states(
    new_log_mu,
    new_p_binom,
    new_log_mu_errors,
    new_p_binom_errors,
    pred_cnv,
    max_allele_copy=5,
    max_total_copy=6,
    n_sigma=1.0,
    min_prop_threshold=0.1,  # TODO
    EPS_BAF=0.05,  # TODO
):
    n_states = len(new_log_mu)
    mu = np.exp(new_log_mu)

    """
    idx_diploid_normal = find_diploid_balanced_state(
        new_log_mu,
        new_p_binom,
        pred_cnv,
        min_prop_threshold=min_prop_threshold,
        EPS_BAF=EPS_BAF,
    )

    scalefactor = 2.0 / mu[idx_diploid_normal]
    """
    scalefactor = 1.0
    candidates = np.array(
        [
            [i, j]
            for i in range(max_allele_copy + 1)
            for j in range(max_allele_copy + 1)
            if (not (i == 0 and j == 0)) and (i + j <= max_total_copy)
        ]
    )

    logger.info(
        f"Filtering {len(candidates)} candidate ACN states for consistency with measurements"
    )

    consistent_states_baf, consistent_states_both = {}, {}

    for s in range(n_states):
        consistent_baf, consistent_both = [], []
        obs_baf, obs_rdr = new_p_binom[s], mu[s]

        # baf_std = np.sqrt(baf_var_per_state[s])
        # rdr_std = np.sqrt(rdr_var_per_state[s])
        baf_std = new_p_binom_errors[s]
        rdr_std = mu[s] * new_log_mu_errors[s]

        for A, B in candidates:
            total = A + B

            exp_baf = A / total if total > 0 else np.nan
            exp_rdr = total / scalefactor

            baf_consistent = np.abs(obs_baf - exp_baf) <= n_sigma * baf_std
            rdr_consistent = np.abs(obs_rdr - exp_rdr) <= n_sigma * rdr_std

            if baf_consistent:
                consistent_baf.append((A, B))

            if baf_consistent and rdr_consistent:
                consistent_both.append((A, B))

        consistent_states_baf[s] = consistent_baf
        consistent_states_both[s] = consistent_both

        logger.info(
            f"State {s}: obs_baf={obs_baf:.3f}±{baf_std:.3f}, obs_rdr={obs_rdr:.3f}±{rdr_std:.3f} "
            f"-> BAF consistent={consistent_baf}, RDR-BAF consistent={consistent_both}"
        )

    return consistent_states_baf, consistent_states_both


def test_filter_consistent_acn_states():
    log_mu = [
        -0.1447503693281741,
        -0.4559718905828648,
        0.375514054163467,
        0.08874972645718063,
        -0.16125858957680456,
        0.46714026176291773,
        -2.4404262483049046,
        1.2395390024166972,
    ]
    log_mu_errors = [
        0.024178389299085976,
        0.08852660326264564,
        0.10548528206909513,
        0.06535499147384073,
        0.09234353070930724,
        0.06127245825343991,
        0.07128597075848096,
        0.030963634035987186,
    ]

    p_binom = [
        0.5012499737991472,
        0.08023295254316322,
        0.31270148795265096,
        0.17312254870436608,
        0.38084808790550945,
        0.46782630226580574,
        0.13744771136181827,
        1458.1475274694344]
    
    p_binom_errors = [
        0.0006507893332657055,
        0.0025121063532950005,
        0.0028110796109445174,
        0.003472637803974673,
        0.007735573740410467,
        0.002282115125666023,
        0.019953170089816126,
        240.05274911055483
    ]

    # Dummy pred_cnv
    pred_cnv = [0] * len(log_mu)

    consistent_states_baf, consistent_states_both = filter_consistent_acn_states(
        np.array(log_mu),
        np.array(p_binom),
        np.array(log_mu_errors),
        np.array(p_binom_errors),
        pred_cnv,
    )

    print("Consistent states BAF:", consistent_states_baf)
    print("Consistent states Both:", consistent_states_both)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_filter_consistent_acn_states()
