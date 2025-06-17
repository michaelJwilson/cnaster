import glob
import json
import numpy as np
from cnaster_rs import ellipse

# TODO HACK
DEFAULT_PHY_ID = 1

class Clone:
    def __init__(self, fpath, x0=None):
        with open(fpath, "r") as f:
            data = json.load(f)

        center = np.array(data["center"]).reshape(2, 1)
        L = np.array(data["L"])

        if x0 is not None:
            x0 = x0.reshape(2, 1)
            center += x0

        self.id = data["id"]
        self.cnas = data["cnas"]
        self.ellipse = ellipse.CnaEllipse(center, L)

    def __repr__(self):
        return f"<Clone ellipse={self.ellipse} cnas={len(self.cnas)}>"
    
def get_clones(config, phy_id=DEFAULT_PHY_ID):
    """
    Load clones from phylogeny files.
    """
    x0 = np.array([0.5, 0.5]).reshape(2, 1)

    clones = [
        Clone(xx, x0=x0)
        for xx in sorted(
            glob.glob(config.output_dir + f"/phylogenies/phylogeny{phy_id}/*.json")
        )
    ]

    return clones

def query_clones(config, clones, x, y, z):
    # NB find the corresponding clone.
    query = np.array([x, y]).reshape(2, 1)
    query /= config.phylogeny.spatial_scale

    isin = [clone.ellipse.contains(query) for clone in clones]
    candidates = [clone for clone, inside in zip(clones, isin) if inside]

    # NB we choose the smallest of overlapping ellipse as a (close) proxy for later evolved.
    if candidates:
        return min(candidates, key=lambda c: c.ellipse.det_l)
    else:
        return None
    
def get_cnas(config, clone_ids, phy_id=DEFAULT_PHY_ID):
    """
    Load CNAs from phylogeny files.
    """
    clones = get_clones(config, phy_id=phy_id)

    return [clones[clone_id -1].cnas for clone_id in clone_ids if clone_id != -1]

def construct_frac_cnas(num_segments, segment_size_kbp, tumor_purity, cnas):
    rdrs = np.ones(num_segments, dtype=float)
    bafs = 0.5 * np.ones(num_segments, dtype=float)

    # TODO CNA start, end.
    for cna in cnas:
        pos_idx = int(np.floor(cna[1] / segment_size_kbp))
        state = cna[0]

        mat_copy, pat_copy = [int(xx) for xx in cna[0].split(",")]

        rdr = (mat_copy + pat_copy) / 2
        baf = min([mat_copy, pat_copy]) / (mat_copy + pat_copy)

        rdrs[pos_idx] = (1. - tumor_purity) + (rdr * tumor_purity)
        bafs[pos_idx] = 0.5 * (1. - tumor_purity) + tumor_purity * baf * rdr

    return rdrs, bafs
