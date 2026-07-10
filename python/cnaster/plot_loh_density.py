import logging
from typing import Any, Dict, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

'''
def plot_loh_density(
    coords: np.ndarray,
    single_X: np.ndarray,
    single_total_bb_RD: np.ndarray,
    res_combine: Dict[str, Any],
    lengths: Optional[np.ndarray] = None,
    smooth_sigma: float = 0,
    max_alpha: float = 0.5,
    gamma: float = 1.0,
    color: str = "#0055ff",  # Deep blue
    plot_type: str = "both", # "empirical", "model", or "both"
):
    """
    Plots a 3D volumetric density cloud using INVERSE Loss of Heterozygosity (LOH).
    Normal = Opaque (1.0). LOH = Transparent (0.0).
    """
    logger.info("Solving for loh density to render.")
    
    n_bins, _, n_spots = single_X.shape

    # 1. RIGOROUS NaN SCRUBBING
    with np.errstate(divide="ignore", invalid="ignore"):
        baf_raw = single_X[:, 1, :] / single_total_bb_RD

    # Any 0-coverage or NaN data becomes 0.5 (Perfectly Normal/Heterozygous)
    invalid_mask = (single_total_bb_RD == 0) | np.isnan(baf_raw)
    baf_raw[invalid_mask] = 0.5 
    
    # maf=0.0 is loh=1.0, maf=0.5 is loh=0.0
    maf_raw = np.minimum(baf_raw, 1.0 - baf_raw)
    loh_empirical = 1.0 - (2.0 * maf_raw)
    
    if smooth_sigma > 0:
        loh_empirical = gaussian_filter1d(loh_empirical, sigma=smooth_sigma, axis=0)
    
    loh_empirical = np.clip(loh_empirical, 0.0, 1.0)
    # Final safeguard against Gaussian filter NaN propagation
    loh_empirical = np.nan_to_num(loh_empirical, nan=0.0)

    # 2. INFERRED MODEL EQUIVALENT
    loh_model = np.zeros((n_bins, n_spots))
    
    assignments = res_combine["new_assignment"]
    pred_cnv = res_combine["pred_cnv"]
    p_binom = res_combine["new_p_binom"]
    
    n_clones = p_binom.shape[1]
    is_concatenated = (pred_cnv.ndim == 1)
    
    for c in range(n_clones):
        spot_mask = (assignments == c)
        if not np.any(spot_mask):
            continue
            
        if is_concatenated:
            c_pred = pred_cnv[c * n_bins : (c + 1) * n_bins]
        else:
            c_pred = pred_cnv[:, c]
            
        c_p = p_binom[c_pred, c if p_binom.shape[1] > 1 else 0]
        c_maf = np.minimum(c_p, 1.0 - c_p)
        c_loh = 1.0 - (2.0 * c_maf)
        
        loh_model[:, spot_mask] = c_loh[:, None]

    loh_model = np.nan_to_num(loh_model, nan=0.0)

    # 3. 3D MESHGRID
    Z_mesh, S_mesh = np.meshgrid(np.arange(n_bins), np.arange(n_spots), indexing="ij")
    
    X_flat = coords[S_mesh.flatten(), 0]
    Y_flat = -coords[S_mesh.flatten(), 1]
    Z_flat = Z_mesh.flatten()

    base_rgb = mcolors.to_rgb(color)
    
    def build_rgba(loh_signal: np.ndarray) -> np.ndarray:
        """Converts LOH signal into an RGBA array. Normal = Opaque, LOH = Transparent."""
        rgba = np.zeros((n_bins * n_spots, 4))
        rgba[:, :3] = base_rgb

        # INVERSE LOGIC: Normal (loh=0) -> alpha=1.0. LOH (loh=1) -> alpha=0.0
        alpha_channel = max_alpha * (1.0 - (loh_signal.flatten() ** gamma))
        rgba[:, 3] = np.clip(alpha_channel, 0.0, 1.0)
        
        return np.nan_to_num(rgba, nan=0.0)

    # 4. RENDER
    n_plots = 2 if plot_type == "both" else 1
    fig = plt.figure(figsize=(10 * n_plots, 10), dpi=300, facecolor="white")
    
    titles, rgba_arrays = [], []
    
    if plot_type in ["empirical", "both"]:
        rgba_arrays.append(build_rgba(loh_empirical))
    if plot_type in ["model", "both"]:
        rgba_arrays.append(build_rgba(loh_model))

    for i, rgba in enumerate(rgba_arrays):
        ax = fig.add_subplot(1, n_plots, i + 1, projection="3d")
        
        # PRE-FILTER: Drop points that are virtually transparent.
        # This prevents Matplotlib 3D from crashing on millions of invisible points.
        visible_mask = rgba[:, 3] > 0.01

        ax.scatter(
            X_flat[visible_mask], 
            Y_flat[visible_mask], 
            Z_flat[visible_mask], 
            c=rgba[visible_mask], 
            s=1.0, 
            edgecolors="none",
            rasterized=True
        )

        ax.set_xlabel(r"$x$", labelpad=10)
        ax.set_ylabel(r"$y$", labelpad=10)
        ax.set_zlabel(r"$g$", labelpad=10)
        
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.grid(False)

        if lengths is not None:
            chr_boundaries = np.cumsum(lengths)
            for boundary in chr_boundaries[:-1]:
                ax.plot(
                    [X_flat.min(), X_flat.max()], 
                    [Y_flat.max(), Y_flat.max()], 
                    [boundary, boundary], 
                    color="gray", linewidth=0.5, alpha=0.5, linestyle="--"
                )

    fig.tight_layout()
    return fig
'''


def nan_gaussian_filter1d(data, sigma, fill_value=0.0):
    valid_mask = ~np.isnan(data)

    data_zeroed = np.copy(data)
    data_zeroed[~valid_mask] = 0.0

    smoothed_data = gaussian_filter1d(data_zeroed, sigma=sigma)
    smoothed_mask = gaussian_filter1d(valid_mask.astype(float), sigma=sigma)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = smoothed_data / smoothed_mask

    result[np.isnan(result)] = fill_value

    return result


def plot_loh_density(
    coords: np.ndarray,
    single_X: np.ndarray,
    single_total_bb_RD: np.ndarray,
    res_combine: Dict[str, Any],
    lengths: Optional[np.ndarray] = None,
    smooth_sigma: float = 0.1,
    max_alpha: float = 0.5,
    gamma: float = 1.0,
    color: str = "#0055ff",  # Deep blue
    plot_type: str = "both",  # "empirical", "model", or "both"
):
    """
    Plots a 3D volumetric density cloud of Loss of Heterozygosity (LOH).
    Pure LOH = Opaque blue (at max_alpha).
    Normal/Diploid or Missing Data = Transparent (dropped to prevent overloading).
    """
    logger.info("Solving for loh density to render.")

    n_bins, _, n_spots = single_X.shape

    # 1. RIGOROUS NaN SCRUBBING
    with np.errstate(divide="ignore", invalid="ignore"):
        baf_raw = single_X[:, 1, :] / single_total_bb_RD

    baf_smooth = nan_gaussian_filter1d(baf_raw, smooth_sigma, fill_value=0.5)

    maf_raw = np.minimum(baf_smooth, 1.0 - baf_smooth)
    loh_empirical = 1.0 - (2.0 * maf_raw)

    loh_empirical = np.clip(loh_empirical, 0.0, 1.0)
    loh_empirical = np.nan_to_num(loh_empirical, nan=0.0)

    # 2. INFERRED MODEL EQUIVALENT
    loh_model = np.zeros((n_bins, n_spots))

    assignments = res_combine["new_assignment"]
    pred_cnv = res_combine["pred_cnv"]
    p_binom = res_combine["new_p_binom"]

    n_clones = len(np.unique(assignments))
    is_concatenated = pred_cnv.ndim == 1

    for c in range(n_clones):
        spot_mask = assignments == c
        if not np.any(spot_mask):
            continue

        if is_concatenated:
            c_pred = pred_cnv[c * n_bins : (c + 1) * n_bins]
        else:
            c_pred = pred_cnv[:, c]

        c_p = p_binom[c_pred, c if p_binom.shape[1] > 1 else 0]
        c_maf = np.minimum(c_p, 1.0 - c_p)
        c_loh = 1.0 - (2.0 * c_maf)

        loh_model[:, spot_mask] = c_loh[:, None]

    loh_model = np.nan_to_num(loh_model, nan=0.0)

    Z_mesh, S_mesh = np.meshgrid(np.arange(n_bins), np.arange(n_spots), indexing="ij")

    X_flat = coords[S_mesh.flatten(), 0]
    Y_flat = -coords[S_mesh.flatten(), 1]
    Z_flat = Z_mesh.flatten()

    base_rgb = mcolors.to_rgb(color)

    def build_rgba(loh_signal: np.ndarray) -> np.ndarray:
        rgba = np.zeros((n_bins * n_spots, 4))
        rgba[:, :3] = base_rgb
        # alpha_channel = max_alpha * (1.0 - (loh_signal.flatten() ** gamma))
        alpha_channel = max_alpha * loh_signal.flatten() ** gamma
        rgba[:, 3] = np.clip(alpha_channel, 0.0, 1.0)
        return np.nan_to_num(rgba, nan=0.0)

    # 4. RENDER
    n_plots = 2 if plot_type == "both" else 1
    fig = plt.figure(figsize=(10 * n_plots, 10), dpi=300, facecolor="white")

    titles, rgba_arrays = [], []

    if plot_type in ["empirical", "both"]:
        rgba_arrays.append(build_rgba(loh_empirical))
    if plot_type in ["model", "both"]:
        rgba_arrays.append(build_rgba(loh_model))

    for i, rgba in enumerate(rgba_arrays):
        ax = fig.add_subplot(1, n_plots, i + 1, projection="3d")

        visible_mask = rgba[:, 3] > 0.01

        # NB: Axes Re-mapped!
        # Z_flat (Genomic) is plotted on the X axis.
        # X_flat, Y_flat (Spatial) are plotted on the Y and Z axes.
        ax.scatter(
            Z_flat[visible_mask],
            X_flat[visible_mask],
            Y_flat[visible_mask],
            c=rgba[visible_mask],
            s=1.0,
            edgecolors="none",
            rasterized=True,
        )

        # Force the new X axis (Genomic g) to be twice as long as the Y/Z spatial axes
        ax.set_box_aspect((2, 1, 1))

        # Rotate camera: Elevates slightly, and azimuth -45 positions the Y-Z plane
        # (which now contains our spatial x,y data) squarely on the back-left wall.
        ax.view_init(elev=20, azim=-70)

        # Labels updated to reflect the swapped data mapping
        ax.set_xlabel(r"$g$", labelpad=10)
        ax.set_ylabel(r"$x$", labelpad=10)
        ax.set_zlabel(r"$y$", labelpad=10)

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.grid(False)

        """
        if lengths is not None:
            chr_boundaries = np.cumsum(lengths)
            for boundary in chr_boundaries[:-1]:
                # Boundary lines re-mapped to match the new plot axes
                ax.plot(
                    [boundary, boundary], 
                    [X_flat.min(), X_flat.max()], 
                    [Y_flat.max(), Y_flat.max()], 
                    color="gray", linewidth=0.5, alpha=0.5, linestyle="--"
                )
        """
    fig.tight_layout()
    return fig
