import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


def merge_dicts(first, second):
    merged = first.copy()
    collision = False

    for k, v in second.items():
        if k in merged:
            collision = True
            logger.warning(
                f"Key clash on '{k}': overwriting value {merged[k]} with {v}"
            )
        merged[k] = v

    if not collision:
        logger.info(f"Safely merged dictionaries with no collisions.")

    return merged


def write_tsv(opath, df=None, header=True, index=False, index_label=None):
    if df is None:
        df = pd.DataFrame()

    logger.info(f"Writing to {opath}.")

    df.to_csv(opath, sep="\t", header=header, index=index, index_label=index_label)


def write_fig(opath, fig=None, transparent=True, bbox_inches="tight"):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    logger.info(f"Writing figure to {opath}.")
    fig.savefig(opath, format="pdf", transparent=transparent, bbox_inches=bbox_inches)
