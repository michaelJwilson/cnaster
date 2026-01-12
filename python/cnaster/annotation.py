import numpy as np
import pandas as pd
from cnaster.config import get_global_config, start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


def get_clone_label_annotation(config=None):
    if config is None:
        config = get_global_config()

    logger.warning(f"Assuming known clone labels={config.annotation.clone_label}")

    clone_id = (
        pd.read_csv(config.annotation.clone_label, sep="\t", index_col=0)["labels"]
        .str.replace("clone_", "")
        .str.replace("normal", "-1")
        .astype(int)
        .to_numpy()
    )
    clone_id += 1

    initial_clone_index_baf = [
        np.where(clone_id == xx)[0] for xx in np.unique(clone_id)
    ]

    known_rdr_normal = np.sum(single_X[:, 0, (clone_id == 0)], axis=1)

    bidx_inconfident = np.where(
        known_rdr_normal < config.quality.min_normal_count_perbin
    )[0]
    known_rdr_normal[bidx_inconfident] = 0

    # NB normalized.
    known_rdr_normal = known_rdr_normal / np.sum(known_rdr_normal)

    spots_coverage = np.sum(single_X[:, 0, :], axis=0)

    known_single_base_nb_mean = known_rdr_normal.reshape(
        -1, 1
    ) @ spots_coverage.reshape(1, -1)

    return initial_clone_index_baf, known_single_base_nb_mean
