import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_filter_genes(filter_gene_file):
    filter_gene_list = pd.read_csv(filter_gene_file, header=None)
    filter_gene_list = set(list(filter_gene_list.iloc[:, 0]))

    return filter_gene_list


def get_filter_ranges(filter_range_file):
    ranges = pd.read_csv(
        filter_range_file, header=None, sep="\t", names=["Chrname", "Start", "End"]
    )

    if "chr" in ranges.Chrname.iloc[0]:
        ranges["CHR"] = [int(x[3:]) for x in ranges.Chrname.values]
    else:
        ranges.rename(columns={"Chrname": "Chr"}, inplace=True)

    ranges.sort_values(by=["Chr", "Start"], inplace=True)

    return ranges
