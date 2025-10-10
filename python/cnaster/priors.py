import numpy as np
from functools import lru_cache
from cnaster.integer_copy import get_ordered_acn
from sklearn.mixture import GaussianMixture


def get_rdr_prior_centers():
    rdrs = [1.0]

    for a, b in get_ordered_acn():
        if a == b == 1:
            continue

        rdrs.append((a + b) / 2.0)

    rdrs = np.array(rdrs)
    rdrs, _ = np.unique(rdrs, return_counts=True)

    return rdrs


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
    bafs, _ = np.unique(bafs, return_counts=True)

    return bafs


def get_rdr_prior_sigma():
    return 0.1


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


# New: RDR prior analogous to BAF prior
@lru_cache(maxsize=32)
def get_rdr_prior(sigma=None):
    if sigma is None:
        sigma = get_rdr_prior_sigma()

    centers = get_rdr_prior_centers().reshape(-1, 1)
    n_comp = centers.shape[0]

    gm = GaussianMixture(n_components=n_comp, covariance_type="diag")
    gm.weights_ = np.ones(n_comp) / n_comp
    gm.means_ = centers.copy()

    gm.covariances_ = np.full((n_comp, 1), float(sigma) ** 2)
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


# New: RDR evaluation, normalized at RDR = 1.0
def rdr_prior_eval(x, sigma=None):
    if sigma is None:
        sigma = get_rdr_prior_sigma()

    gm = get_rdr_prior(sigma=sigma)
    x_arr = np.asarray(x, dtype=float).reshape(-1, 1)

    ln_p = gm.score_samples(x_arr)
    ln_p0 = gm.score_samples(np.asarray([1.0], dtype=float).reshape(-1, 1))[0]

    return ln_p - ln_p0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # BAF data
    centers = get_baf_prior_centers()
    sigma_baf = get_baf_prior_sigma()
    x = np.linspace(0.0, 1.0, 1001)
    lnprob = baf_prior_eval(x, sigma=sigma_baf)
    pdf = np.exp(lnprob)

    # RDR data
    centers_rdr = get_rdr_prior_centers()
    sigma_rdr = get_rdr_prior_sigma()
    x2 = np.linspace(0.0, max(centers_rdr.max() + 1.0, 3.0), 1001)
    lnprob2 = rdr_prior_eval(x2, sigma=sigma_rdr)
    pdf2 = np.exp(lnprob2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # BAF subplot
    ax = axes[0]
    ax.plot(x, pdf, label=None, lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("BAF")
    ax.set_ylabel(r"$\tilde p$(BAF)")
    ax.set_title(f"Gaussian mixture BAF prior (sigma={sigma_baf})")
    ax.legend(frameon=False)

    # RDR subplot
    ax2 = axes[1]
    ax2.plot(x2, pdf2, label=None, lw=1)
    ax2.set_xlim(0, x2.max())
    ax2.set_xlabel("RDR")
    ax2.set_ylabel(r"$\tilde p$(RDR)")
    ax2.set_title(f"Gaussian mixture RDR prior (sigma={sigma_rdr})")
    ax2.legend(frameon=False)

    fig.tight_layout()
    plt.show()
