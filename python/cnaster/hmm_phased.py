from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.hmm_sitewise import (
    compute_emission_probability_nb_betabinom_phased,
    forward_marginalize_phased,
    backward_marginalize_phased,
)


class hmm_phased(hmm_nophasing):
    def __init__(self, params="stmp", t=1 - 1e-4):
        super().__init__(params=params, t=t)

    @staticmethod
    def compute_emission_probability_nb_betabinom(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
    ):
        return compute_emission_probability_nb_betabinom_phased(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )

    @staticmethod
    def forward_lattice(
        lengths,
        log_transmat,
        log_startprob,
        log_emission,
        log_sitewise_transmat,
    ):
        return forward_marginalize_phased(
            lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
        )

    @staticmethod
    def backward_lattice(
        lengths,
        log_transmat,
        log_startprob,
        log_emission,
        log_sitewise_transmat,
    ):
        return backward_marginalize_phased(
            lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
        )
