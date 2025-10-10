import numpy as np
from functools import lru_cache
from cnaster.integer_copy import get_ordered_acn
from sklearn.mixture import GaussianMixture


def get_baf_prior_centers():
    bafs = [0.0]

    for a, b in get_ordered_acn():
        if a == b == 0:
            continue

        baf = b / (a + b)
        mirrored_baf = 1.0 - baf

        bafs += [baf, mirrored_baf]

    bafs = np.array(bafs)

    # NB sorted;
    bafs, cnts = np.unique(bafs, return_counts=True)

    return bafs


def get_baf_prior_sigma():
    return 0.01


@lru_cache(maxsize=32)
def get_baf_prior(sigma=None):
    # Use centralized sigma if not provided
    if sigma is None:
        sigma = get_baf_prior_sigma()

    centers = get_baf_prior_centers().reshape(-1, 1)
    n_comp = centers.shape[0]

    gm = GaussianMixture(n_components=n_comp, covariance_type="diag")
    gm.weights_ = np.ones(n_comp) / n_comp
    gm.means_ = centers.copy()

    # diagonal covariances: shape (n_components, n_features) -> (n_comp, 1)
    gm.covariances_ = np.full((n_comp, 1), float(sigma) ** 2)

    # precisions_cholesky_ expected shape matches covariances_ for 'diag'
    gm.precisions_cholesky_ = 1.0 / np.sqrt(gm.covariances_)

    return gm


def baf_prior_eval(x, sigma=None):
    # Use centralized sigma if not provided
    if sigma is None:
        sigma = get_baf_prior_sigma()

    gm = get_baf_prior(sigma=sigma)
    x_arr = np.asarray(x, dtype=float).reshape(-1, 1)

    # score_samples returns ln p(x). Subtract ln p(0.5) so exp(result) == p(x)/p(0.5)
    ln_p = gm.score_samples(x_arr)
    ln_p0 = gm.score_samples(np.asarray([0.5], dtype=float).reshape(-1, 1))[0]

    return ln_p - ln_p0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    baf_prior = get_baf_prior_centers()

    centers = get_baf_prior_centers()

    # use centralized sigma for plotting / evaluation
    sigma = get_baf_prior_sigma()
    x = np.linspace(0.0, 1.0, 1001)
    lnprob = baf_prior_eval(x, sigma=sigma)
    pdf = np.exp(lnprob)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, pdf, label=None, lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("BAF")
    ax.set_ylabel("p(BAF)")
    ax.set_title(r"Gaussian mixture BAF prior ($\sigma=0.5$)")
    ax.legend(frameon=False)

    fig.tight_layout()

    plt.show()
