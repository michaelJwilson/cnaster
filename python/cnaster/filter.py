import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_filter_genes(filter_gene_file):
    return pd.read_csv(filter_gene_file, header=None)


def get_filter_ranges(filter_range_file):
    ranges = pd.read_csv(
        filter_range_file, header=None, sep="\t", names=["Chr", "Start", "End"]
    )

    if "chr" in ranges.Chr.iloc[0]:
        ranges["Chr"] = [int(x[3:]) for x in ranges.Chr.values]

    ranges.sort_values(by=["Chr", "Start"], inplace=True)

    return ranges
