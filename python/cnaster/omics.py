from collections import namedtuple

import numpy as np
import pandas as pd

from cnaster.config import start_time
from cnaster.logger import get_logger
from cnaster.recomb import get_sitewise_transmat
from cnaster.reference import get_reference_genes
from cnaster.spatio_genomic_counts import SpatioGenomicCounts
from cnaster.utils import cacher

logger = get_logger(__name__, start_time=start_time)


def greedy_binning_nobreak(
    block_lengths,
    block_umi,
    block_snp_umi,
    block_normal_umi,
    secondary_min_umi,
    secondary_min_snp_umi,
    secondary_min_normal_umi,
    max_binlength,
):
    """
    Given a set of blocks, find new bins that meet requirements on:
    - minimum total UMIs
    - minimum SNP-covering UMIs
    - minimum normal UMIs
    - maximum bin length
    """
    assert (
        len(block_lengths)
        == len(block_umi)
        == len(block_snp_umi)
        == len(block_normal_umi)
    ), (
        f"Block array length mismatch: "
        f"lengths={len(block_lengths)}, "
        f"umi={len(block_umi)}, "
        f"snp_umi={len(block_snp_umi)}, "
        f"normal_umi={len(block_normal_umi)}"
    )

    # NB (start, end) indices of new bins that aggregate old blocks to
    #    meet umi, length, etc. requirements.
    bin_ranges = []
    s = 0

    while s < len(block_lengths):
        t = s + 1

        # NB extend included blocks until meets required umi count.
        while t < len(block_lengths):
            total_umi = np.sum(block_umi[s:t])
            snp_umi = np.sum(block_snp_umi[s:t])
            normal_umi = np.sum(block_normal_umi[s:t])
            length = np.sum(block_lengths[s:t])

            # NB check if all min requirements are met
            meets_umi = total_umi >= secondary_min_umi
            meets_snp = snp_umi >= secondary_min_snp_umi
            meets_normal = normal_umi >= secondary_min_normal_umi
            all_criteria_met = meets_umi and meets_snp and meets_normal

            # NB break if bin is too long but meets UMI requirements
            if length >= max_binlength and all_criteria_met:
                logger.warning(
                    f"Solved for bin length={length/max_binlength:>6.2f} [max_binlength] "
                    f"(umi={total_umi:>8}, snp-umi={snp_umi:>8}, normal-umi={normal_umi:>8})"
                )
                t = max(t - 1, s + 1)
                break

            # NB continue if criteria not met and not too long
            if all_criteria_met:
                break

            t += 1

        # NB final counts for bin [s:t]
        total_umi = np.sum(block_umi[s:t])
        snp_umi = np.sum(block_snp_umi[s:t])
        normal_umi = np.sum(block_normal_umi[s:t])
        length = np.sum(block_lengths[s:t])

        # NB check if it's a small bin at the end that doesn't meet criteria
        if s > 0 and t == len(block_lengths):
            if (
                total_umi < secondary_min_umi
                or snp_umi < secondary_min_snp_umi
                or normal_umi < secondary_min_normal_umi
            ):
                logger.debug(
                    f"Last bin failed thresholds "
                    f"(UMI={total_umi:>8}/{secondary_min_umi:<8}, "
                    f"SNP-UMI={snp_umi:>8}/{secondary_min_snp_umi:<8}, "
                    f"normal-UMI={normal_umi:>8}/{secondary_min_normal_umi:<8}), "
                    f"merging with previous."
                )
                bin_ranges[-1][1] = t
            else:
                bin_ranges.append([s, t])
        else:
            bin_ranges.append([s, t])

        s = t

    bin_ids = np.zeros(len(block_lengths), dtype=int)

    for i, x in enumerate(bin_ranges):
        bin_ids[x[0] : x[1]] = i

    # NB return new bin ids for each block, where new bins meet umi, length, etc. requirements.
    return bin_ids


# TODO assumes reference gene contains all those present in Visium anndata.
@cacher("gene_snp_table.tsv")
def form_gene_snp_table(
    unique_snp_ids,
    hgtable_file,
    adata,
    verbose=False,
    num_preceeding_rows=100,  # MAGIC
):
    logger.info(f"Forming gene & snp meta data.")

    # NB includes both gene and SNP info: CHR, START, END, snp_id, gene, is_interval
    df_gene = get_reference_genes(hgtable_file)

    logger.info(f"Filtering reference genes to those in visium.")

    common_genes = set(df_gene.gene) & set(adata.var.index)
    genes_not_in_reference = set(adata.var.index) - common_genes

    logger.info(
        f"Found {100. * len(common_genes) / len(adata.var.index):.2f}% of visium genes to be in reference."
    )

    # NB enriched for sex-chromosome and mitochondrial genes, many lncRNAs/antisense/pseudogenes, and CT antigens commonly over-expressed in tumors.
    logger.info(
        f"Found {len(genes_not_in_reference):_} genes to be in visium but not in reference:"
    )

    genes_sorted = sorted(genes_not_in_reference)

    if verbose:
        for i in range(0, len(genes_sorted), 10):
            chunk = genes_sorted[i : i + 10]
            logger.info(", ".join(chunk))

    # NB limits reference genes to those present in (filtered) AnnData UMIs.
    df_gene = df_gene[df_gene.gene.isin(adata.var.index)]

    # NB add SNP info: {contig}_{pos}_{ref}_{alt}.
    snp_chr = np.array([int(x.split("_")[0]) for x in unique_snp_ids])
    snp_pos = np.array([int(x.split("_")[1]) for x in unique_snp_ids])
    snp_end = snp_pos + 1

    # NB vertical concatenation
    df_gene_snp = pd.concat(
        [
            df_gene,
            pd.DataFrame(
                {
                    "CHR": snp_chr,
                    "START": snp_pos,
                    "END": snp_end,
                    "snp_id": unique_snp_ids,
                    "gene": None,
                    "is_interval": False,
                }
            ),
        ],
        ignore_index=True,
    )

    logger.debug(f"Sorting df_gene_snp")

    df_gene_snp.sort_values(by=["CHR", "START"], inplace=True)

    logger.debug(f"Assigning genes to SNPs")

    """
    Assigns genes to each SNP:  for each SNP (with not null snp_id), find the previous gene (is_interval == True)
    such that the SNP start position is within the gene start & end interval.
    """

    # NB == is_gene
    vec_is_interval = df_gene_snp.is_interval.to_numpy()

    vec_chr = df_gene_snp.CHR.to_numpy()
    vec_start = df_gene_snp.START.to_numpy()
    vec_end = df_gene_snp.END.to_numpy()

    # NB loops over sites.
    for i in np.where(df_gene_snp.gene.isnull())[0]:
        # TODO first SNP has no gene.
        if i == 0:
            continue

        this_pos = vec_start[i]

        # NB look for an overlapping gene, closest in START, in the previous {num_preceeding_rows} rows (on same contig).
        j = i - 1

        # NB assigns closest in start.
        while j >= 0 and j >= (i - num_preceeding_rows) and (vec_chr[j] == vec_chr[i]):
            if (
                vec_is_interval[j]
                and vec_start[j] <= this_pos
                and vec_end[j] > this_pos
            ):
                df_gene_snp.iloc[i, 4] = df_gene_snp.iloc[j]["gene"]
                break

            j -= 1

    logger.debug(f"Assigned SNPs to genes.")

    # NB remove SNPs that have no corresponding genes.
    isin = ~df_gene_snp.gene.isnull()

    # TODO retaining 84.623% of SNPs with known gene (given Gencode filtered by AnnData) for num_preceeding_rows=50.
    logger.info(
        f"Retaining {100.0 * np.mean(isin[~df_gene_snp.is_interval]):.3f}% of snps with matched gene (given reference filtered by visium panel) for num_preceeding_rows={num_preceeding_rows}."
    )

    logger.info(
        f"Failed to find overlapping gene for:\n{df_gene_snp[df_gene_snp.gene.isnull()]}"
    )

    df_gene_snp = df_gene_snp[isin]

    logger.info(f"Created gene-snp query table:\n{df_gene_snp.head()}")

    return df_gene_snp


def binned_gene_snp(df_gene_snp, key="bin_id"):
    # NB table with contig range + set of genes + snp_ids,
    #    only for those defined with key bin_id.
    table_bininfo = (
        # df_gene_snp[~df_gene_snp.bin_id.isnull()]
        df_gene_snp[~getattr(df_gene_snp, key).isnull()]
        .groupby(key)
        .agg(
            {
                "CHR": "first",
                "START": "first",
                "END": "last",
                "gene": set,
                "snp_id": set,
            }
        )
        .reset_index()  # TBC (0, ..., N-1).
    )
    # table_bininfo["ARM"] = "."
    table_bininfo["INCLUDED_GENES"] = [
        ",".join([x for x in y if not x is None]) for y in table_bininfo.gene.values
    ]
    table_bininfo["INCLUDED_SNP_IDS"] = [
        ",".join([x for x in y if not x is None]) for y in table_bininfo.snp_id.values
    ]
    table_bininfo["NORMAL_COUNT"] = np.nan
    table_bininfo["N_SNPS"] = [
        len([x for x in y if not x is None]) for y in table_bininfo.snp_id.values
    ]

    table_bininfo.drop(columns=["gene", "snp_id"], inplace=True)

    logger.info(
        f"Finalizing genomic segment annotation for {table_bininfo.shape[0]} bins, given initial {len(df_gene_snp.gene.unique())} genes ({table_bininfo.shape[0] / len(df_gene_snp.gene.unique()):.2f} sampling)."
    )

    return table_bininfo


def summarize_blocks(
    gene_snp_table,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
    block_key=None,
    normal_candidates=None,
    sort_key="total_umi",
):
    logger.info(f"Summarizing blocks ...")

    assert block_key is not None, "block_key must be specified"
    assert block_key in gene_snp_table.columns, f"{block_key} not in DataFrame"

    map_snp_index = {x: i for i, x in enumerate(unique_snp_ids)}

    block_summary = gene_snp_table.groupby(block_key).agg(
        num_snps=("snp_id", lambda x: x.notna().sum()),
        num_genes=("is_interval", "sum"),
        genes=("gene", lambda x: list({g for g in x if g is not None})),
        snp_ids=("snp_id", lambda x: [s for s in x if s is not None]),
        chr=("CHR", "first"),
        start=("START", "min"),
        end=("END", "max"),
    )

    # NB Mbp -> Kbp.
    block_summary["length"] = 1_000.0 * (block_summary["end"] - block_summary["start"])

    gene_names = adata.var.index.to_numpy()
    gene_index_map = {g: i for i, g in enumerate(gene_names)}
    count_matrix = adata.layers["count"]  # (n_spots, n_genes)

    total_umis = np.zeros(len(block_summary), dtype=int)
    snp_umis = np.zeros(len(block_summary), dtype=int)

    normal_umis = np.zeros(len(block_summary), dtype=int)
    normal_snp_umis = np.zeros(len(block_summary), dtype=int)

    if normal_candidates is not None:
        assert count_matrix.shape[0] == len(
            normal_candidates
        ), f"{count_matrix.shape[0]} != {len(normal_candidates)}"

        assert cell_snp_Aallele.shape[0] == len(
            normal_candidates
        ), f"{cell_snp_Aallele.shape[0]} != {len(normal_candidates)}"
        assert cell_snp_Ballele.shape[0] == len(
            normal_candidates
        ), f"{cell_snp_Ballele.shape[0]} != {len(normal_candidates)}"

    for idx, (block_id, row) in enumerate(block_summary.iterrows()):
        genes = row["genes"]
        if genes:
            gene_idx = [gene_index_map[g] for g in genes if g in gene_index_map]
            if gene_idx:
                block_sum = count_matrix[:, gene_idx].sum()
                total_umis[idx] = int(block_sum)

                # Calculate normal spot UMIs
                if normal_candidates is not None:
                    normal_umis[idx] = int(
                        count_matrix[normal_candidates, :][:, gene_idx].sum()
                    )

        # SNP-covering UMIs
        snp_ids = row["snp_ids"]
        if snp_ids:
            # TODO HACK?
            snp_idx = np.array([map_snp_index[s] for s in snp_ids if s])
            if len(snp_idx) > 0:
                snp_umis[idx] = int(
                    cell_snp_Aallele[:, snp_idx].sum()
                    + cell_snp_Ballele[:, snp_idx].sum()
                )

                # Calculate SNP-covering UMIs for normal spots
                if normal_candidates is not None:
                    normal_snp_umis[idx] = int(
                        cell_snp_Aallele[np.ix_(normal_candidates, snp_idx)].sum()
                        + cell_snp_Ballele[np.ix_(normal_candidates, snp_idx)].sum()
                    )

    block_summary["total_umi"] = total_umis
    block_summary["snp_umi"] = snp_umis

    block_summary["normal_umi"] = normal_umis
    block_summary["normal_snp_umi"] = normal_snp_umis

    if sort_key is not None:
        block_summary = block_summary.sort_values(sort_key, ascending=False)

    # TODO MAGIC keyword
    max_rows = 25

    logger.info(
        f"Breakdown of genes/snps/umi per {block_key} sorted by {sort_key} (top {max_rows}):"
    )
    logger.info(
        f"{'block id':<10}\t{'chr':>4}\t{'start':>12}\t{'length':>12} [Kbp]\t{'snps':>8}\t{'genes':>8}\t{'total umi':>12}\t{'snp umi':>12}\t{'normal umi':>12}\t{'normal snp umi':>12}"
    )
    logger.info("-" * 136)

    for ii, (block_id, row) in enumerate(block_summary.iterrows()):
        logger.info(
            f"{block_id:<10}\t{row['chr']:>4}\t{row['start']:>12}\t{row['length'] / 1.e6:>12}\t{row['num_snps']:>8}\t{row['num_genes']:>8}\t"
            f"{row['total_umi']:>12}\t{row['snp_umi']:>12}\t{row['normal_umi']:>12}\t{row['normal_snp_umi']:>12}"
        )

        if ii > max_rows:
            break

    logger.info(
        f"\n"
        f"median block length: {block_summary['length'].median() / 1.e6:.1f} [Kbp],\n"
        f"mean block length: {block_summary['length'].mean() / 1.e6:.1f} [Kbp],\n"
        f"median snps/block: {block_summary['num_snps'].median():.1f},\n"
        f"median genes/block: {block_summary['num_genes'].median():.1f},\n"
        f"median umis/block: {block_summary['total_umi'].median():.1f},\n"
        f"median snp-umis/block: {block_summary['snp_umi'].median():.1f},\n"
        f"total blocks: {len(block_summary):_},\n"
        f"total umis: {block_summary['total_umi'].sum():_},\n"
        f"total snp-umis: {block_summary['snp_umi'].sum():_},\n"
        f"total normal umis: {block_summary['normal_umi'].sum():_},\n"
        f"total normal snp-umis: {block_summary['normal_snp_umi'].sum():_},\n"
        f"blocks with 0 umis: {(block_summary['total_umi'] == 0).mean():.1%},\n"
        f"blocks with <100 umis: {(block_summary['total_umi'] < 100).mean():.1%},\n"
        f"blocks with snp-umis, but no gene-umis: {((block_summary['snp_umi'] > 0) & (block_summary['total_umi'] == 0)).mean():.1%}\n"
    )

    if block_summary.index.isna().any():
        logger.warning(f"Found ill-defined group:/n{block_summary.loc[np.nan]}")


# @cacher("blocked_counts.hdf5")
def summarize_counts_for_blocks(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
):
    """
    Aggregates gene-level total UMI counts (from spatial transcriptomics)
    and site-level allele counts (A and B haplotypes from matched SNPs)
    into broader local segments (blocks).
    """
    logger.info(f"Aggregating (snp, umi) counts for genome segmentation.")

    # NB precompute mapping: snp_id -> index
    map_snp_index = {x: i for i, x in enumerate(unique_snp_ids)}

    # NB filter to snps only (drop genes).
    df_snps = df_gene_snp[df_gene_snp.snp_id.notna()].copy()
    df_snps["snp_idx"] = df_snps.snp_id.map(map_snp_index)

    # NB arrays of snp indexs grouped by block_id
    snp_groups = df_snps.groupby("block_id")["snp_idx"].apply(np.array)

    # TODO HACK?  df_gene_snp.gene.notna()
    # NB no repeated genes.
    df_genes = df_gene_snp[df_gene_snp.is_interval == True].copy()
    gene_groups = df_genes.groupby("block_id")["gene"].apply(lambda x: list(set(x)))

    # NB block_ids formed by merging overlapping genes into intervals, merging said intervals
    #    until a threshold min. snp-covering reads and assigning counts to intervals below.
    blocks = df_gene_snp.block_id.unique()
    n_blocks = len(blocks)
    n_spots = adata.shape[0]

    # NB 0 is total umis;  1 index is haplotype 0 counts at each site.
    single_X = np.zeros((n_blocks, 2, n_spots), dtype=int)
    single_base_nb_mean = np.zeros((n_blocks, n_spots))
    single_total_bb_RD = np.zeros((n_blocks, n_spots), dtype=int)

    # precompute gene counts if using sparse matrix (for efficiency)
    gene_counts = adata.layers["count"]  # (n_spots, n_genes)
    gene_names = adata.var.index.to_numpy()

    # TODO numba
    for block_id in blocks:
        # NB BAF/SNPs
        if block_id in snp_groups.index:
            snp_idx = snp_groups[block_id]
            if len(snp_idx) > 0:
                # NB sum haplotype A counts for SNPs in block.
                single_X[block_id, 1, :] = cell_snp_Aallele[:, snp_idx].sum(axis=1)

                # NB sum haplotype A + haplotype B counts for SNPs in block.
                single_total_bb_RD[block_id, :] = cell_snp_Aallele[:, snp_idx].sum(
                    axis=1
                ) + cell_snp_Ballele[:, snp_idx].sum(axis=1)

        # NB RDR/Genes
        if block_id in gene_groups.index:
            genes = gene_groups[block_id]

            # NB genes in df_gene_snp must be present in visium.
            gene_mask = np.isin(gene_names, genes)

            if gene_mask.any():
                single_X[block_id, 0, :] = gene_counts[:, gene_mask].sum(axis=1)

    # NB list of (unique) blocks grouped by contig.
    lengths = df_gene_snp.groupby("CHR")["block_id"].nunique().to_numpy()

    assert single_X.ndim == 3

    return SpatioGenomicCounts(
        lengths, single_X, single_base_nb_mean, single_total_bb_RD
    )


# @cacher("blocked_gene_snp_table.tsv")
def assign_initial_blocks(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
    initial_min_umi,
):
    """
    Initially assigns snps to blocks along the genome, based on merging overlapping gene intervals
    & requiring blocks have a minimm number of snp-covering reads; these can be aggregated up to the
    scale of switch errors in the population-based phasing.

    Returns
    ----------
    df_gene_snp : data frame, names: (CHR, START, END, snp_id, gene, is_interval, block_id)
         Gene and SNP info combined into a single dataframe sorted by (CHR, START).
        "is_interval"=True is a gene, otherwise SNP.
        "gene" contains the name of a gene, or the gene a SNP belongs.
    """
    if "known_id" in df_gene_snp.columns:
        logger.warning(
            "Assuming known genomic segmentation given 'known_id' in df_gene_snp."
        )

        df_gene_snp["known_id"] = df_gene_snp["known_id"].astype("Int64")

        unique_ids = df_gene_snp["known_id"].dropna().unique()
        unique_ids = np.sort(unique_ids) # Ensure order is consistent
        
        mapping = {old_id: i for i, old_id in enumerate(unique_ids)}
        
        df_gene_snp["block_id"] = (
            df_gene_snp["known_id"]
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )
        
        return df_gene_snp

    logger.info(
        f"Creating initial genome segmentation based solely on overlapping genes."
    )

    # NB first level: partition of genome by gene range (if two genes overlap, they are grouped to one range);
    # NB == is_gene.
    is_interval = df_gene_snp.is_interval

    # NB merge overlapping genes.
    tmp_block_genome_intervals = list(
        zip(
            df_gene_snp[is_interval].CHR.to_numpy(),
            df_gene_snp[is_interval].START.to_numpy(),
            df_gene_snp[is_interval].END.to_numpy(),
        )
    )

    # NB (chr, start, end) for first gene.
    first_interval = tmp_block_genome_intervals[0]

    # NB list of intervals for initial block definition (merged overlapping genes).
    block_genome_intervals = [first_interval]
    merged = 0

    # NB called snps are limited to transcripts, ergo limited to genes.
    #
    #    initial intervals are gene ranges merged based on overlap.
    for next_interval in tmp_block_genome_intervals[1:]:
        contig, start, end = next_interval

        # NB check whether overlap with previous block
        if contig == block_genome_intervals[-1][0] and max(
            start, block_genome_intervals[-1][1]
        ) < min(end, block_genome_intervals[-1][2]):
            block_genome_intervals[-1] = (
                contig,
                min(start, block_genome_intervals[-1][1]),
                max(end, block_genome_intervals[-1][2]),
            )

            # TODO warn on excessive length;
            merged += 1
        else:
            block_genome_intervals.append(next_interval)

    # NB TODO 20%?
    logger.info(
        f"Merged {100.0 * merged / len(tmp_block_genome_intervals):.3f}% of genes to ranges as overlapping."
    )

    # NB map block_genome_intervals to block_ranges for rows of df_gene_snp.
    block_ranges = []

    for x in block_genome_intervals:
        # NB overlap of df_gene_snp (genes & sites) with block_genome_interval.
        indexes = np.where(
            (df_gene_snp.CHR.to_numpy() == x[0])
            & (
                np.maximum(df_gene_snp.START.to_numpy(), x[1])
                < np.minimum(df_gene_snp.END.to_numpy(), x[2])
            )
        )[0]

        # index of rows into df_gene_snp that overlap each interval.
        # TODO can fail?
        block_ranges.append((indexes[0], indexes[-1] + 1))

    assert np.all(
        np.array([x[1] for x in block_ranges[:-1]])
        == np.array([x[0] for x in block_ranges[1:]])
    )

    # NB record the initial block id in df_gene_snps
    # BUG previously 0, spuriously assigned to the zeroth block - safe as discarded all snps that don't overlap a gene (merged to block).
    df_gene_snp["initial_block_id"] = -1

    for i, x in enumerate(block_ranges):
        df_gene_snp.iloc[x[0] : x[1], -1] = i

    assert (
        np.all(df_gene_snp["initial_block_id"].values) >= 0
    ), "Found genes/sites with no assigned block."

    logger.info(
        "Assigned snps to initial genome segments (segments == genes, merged on overlap)."
    )

    summarize_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        block_key="initial_block_id",
    )

    logger.info(
        f"Updating genome segmentation to ensure min. snp-covering umi={initial_min_umi} threshold is satisfied for the new segments."
    )

    # NB second level: extend the first level blocks based on haplotype-aggregated counts such that the min. snp-covering umi counts >= initial_min_umi.
    #    maps site_id, {chr}_{pos}_{ref}_{alt} to integer index.
    map_snp_index = {site: index for index, site in enumerate(unique_snp_ids)}
    initial_block_chr = df_gene_snp.CHR.to_numpy()[
        np.array([x[0] for x in block_ranges])
    ]
    block_ranges_new = []
    s = 0

    # NB s is the "lower" initial_block_id and t is the "upper" initial_block_id.
    while s < len(block_ranges):
        t = s

        while t <= len(block_ranges):
            t += 1

            reach_end = t == len(block_ranges)
            change_chr = initial_block_chr[s] != initial_block_chr[t - 1]

            # NB count SNP-covering UMI
            # TODO recalculates for every upper bound.
            involved_snps_ids = df_gene_snp[
                (df_gene_snp.initial_block_id >= s) & (df_gene_snp.initial_block_id < t)
            ].snp_id

            # NB drop genes.
            involved_snps_ids = involved_snps_ids[~involved_snps_ids.isnull()]
            involved_snp_idx = np.array([map_snp_index[x] for x in involved_snps_ids])

            # NB num. of snp-covering umis for initial block ids s to t.
            this_snp_umis = (
                0
                if len(involved_snp_idx) == 0
                else np.sum(cell_snp_Aallele[:, involved_snp_idx])
                + np.sum(cell_snp_Ballele[:, involved_snp_idx])
            )

            if reach_end:
                logger.warning(
                    f"Reached last block with {this_snp_umis}/{initial_min_umi} required snp umis."
                )
                break

            if change_chr:
                t -= 1

                # re-count snp-covering UMIs
                involved_snps_ids = df_gene_snp.snp_id.iloc[
                    block_ranges[s][0] : block_ranges[t - 1][1]
                ]
                involved_snps_ids = involved_snps_ids[~involved_snps_ids.isnull()]

                involved_snp_idx = np.array(
                    [map_snp_index[x] for x in involved_snps_ids]
                )

                this_snp_umis = (
                    0
                    if len(involved_snp_idx) == 0
                    else np.sum(cell_snp_Aallele[:, involved_snp_idx])
                    + np.sum(cell_snp_Ballele[:, involved_snp_idx])
                )

                logger.warning(
                    f"Reached contig end with {this_snp_umis}/{initial_min_umi} required snp umis."
                )

                break

            if this_snp_umis >= initial_min_umi:
                break

        # NB goal is to have assigned this_snp_umis, s and t.
        if (
            this_snp_umis < initial_min_umi
            and s > 0
            and initial_block_chr[s - 1] == initial_block_chr[s]
        ):
            indexes = np.where(df_gene_snp.initial_block_id.isin(np.arange(s, t)))[0]
            block_ranges_new[-1] = (block_ranges_new[-1][0], indexes[-1] + 1)
        else:
            indexes = np.where(df_gene_snp.initial_block_id.isin(np.arange(s, t)))[0]
            block_ranges_new.append((indexes[0], indexes[-1] + 1))

        # NB fast-forward lower block id to upper.
        s = t

    # NB record the block id in df_gene_snp
    df_gene_snp["block_id"] = 0

    for i, x in enumerate(block_ranges_new):
        df_gene_snp.iloc[x[0] : x[1], -1] = i

    logger.info(
        f"Updated genome segmentation given (population phased) genotypes and min. snp-covering umi={initial_min_umi} per segment."
    )

    summarize_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        block_key="block_id",
    )

    return df_gene_snp.drop(columns=["initial_block_id"])


# @cacher("binned_gene_snp_table.tsv")
def create_bin_ranges(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
    single_X,
    single_total_bb_RD,
    refined_lengths,
    secondary_min_umi,
    secondary_min_snp_umi,
    secondary_min_normal_umi,
    normal_candidates=None,
    max_binlength=5e6,
    key="block_id",
):
    """
    Aggregate haplotype blocks to bins with multiple UMI constraints.
    Cannot aggregate blocks separated by __refined_lengths__

    Parameters
    ----------
    df_gene_snp : pd.DataFrame
        Gene and SNP info with block_id assignments.
    adata : AnnData
        Annotated data object.
    cell_snp_Aallele, cell_snp_Ballele : array, (n_spots, n_snps)
        Allele counts.
    unique_snp_ids : list
        SNP identifiers.
    single_X : array, (n_blocks, 2, n_spots)
        Block-level transcript and SNP counts.
    single_total_bb_RD : array, (n_blocks, n_spots)
        Total SNP-covering reads per block.
    refined_lengths : array
        Number of blocks before each phase switch.
    secondary_min_umi : int
        Minimum total UMIs per bin.
    secondary_min_snp_umi : int
        Minimum SNP-covering UMIs per bin.
    secondary_min_normal_umi : int
        Minimum normal-spot UMIs per bin.
    normal_candidates : array-like or None
        Boolean mask or integer indices for normal spots.
    max_binlength : int
        Maximum genomic length per bin.

    Returns
    -------
    df_gene_snp : pd.DataFrame
        Updated with bin_id column.
    """
    logger.info(
        f"Aggregating blocks to bins given baf-inferred phasing to satisfy umi, length, etc. constraints."
    )

    # TODO BUG dropna?
    # NB block intervals: by key (e.g. block_id)
    sorted_chr_pos_both = df_gene_snp.groupby(key).agg(
        {"CHR": "first", "START": "first", "END": "last"}
    )

    # NB block intervals
    block_lengths = (
        sorted_chr_pos_both.END.to_numpy() - sorted_chr_pos_both.START.to_numpy()
    )
    n_blocks = len(block_lengths)

    # NB total umi per block (summed across spots)
    block_umi = np.sum(single_X[:, 0, :], axis=1)

    # NB total snp-covering umi per block
    block_snp_umi = np.sum(single_total_bb_RD, axis=1)

    # NB normal-spot umi per block
    if normal_candidates is not None:
        if (
            isinstance(normal_candidates, (np.ndarray, pd.Series))
            and normal_candidates.dtype == bool
        ):
            normal_idx = np.flatnonzero(normal_candidates)
        else:
            normal_idx = np.asarray(normal_candidates, dtype=int)

        block_normal_umi = np.sum(single_X[:, 0, normal_idx], axis=1)
    else:
        block_normal_umi = np.zeros(n_blocks, dtype=int)
        secondary_min_normal_umi = 0

    assert (
        len(block_lengths)
        == len(block_umi)
        == len(block_snp_umi)
        == len(block_normal_umi)
    ), (
        f"Block array length mismatch: "
        f"lengths={len(block_lengths)}, "
        f"umi={len(block_umi)}, "
        f"snp_umi={len(block_snp_umi)}, "
        f"normal_umi={len(block_normal_umi)}"
    )

    frac_normal = normal_candidates.mean() if normal_candidates is not None else np.nan

    logger.info(
        f"Creating (phased) bin ranges: max_length={max_binlength:_}, "
        f"min_umi={secondary_min_umi}, "
        f"min_snp_umi={secondary_min_snp_umi}, "
        f"min_normal_umi={secondary_min_normal_umi}, "
        f"fraction normal={frac_normal:.3f}"
    )

    # NB breakpoints defined by jump in minor baf, and oversized blocks.
    breakpoints = np.concatenate(
        [
            np.cumsum(refined_lengths),
            np.where(block_lengths > max_binlength)[0],
            np.where(block_lengths > max_binlength)[0] + 1,
        ]
    )

    # NB sorted, unique.
    breakpoints = np.sort(np.unique(breakpoints))

    if breakpoints[0] != 0:
        breakpoints = np.append([0], breakpoints)

    assert np.all(breakpoints[:-1] < breakpoints[1:])

    # NB assign each block to a bin (that meets umi, length, etc. requirements)
    bin_ids = np.zeros(n_blocks, dtype=int)
    offset = 0

    for i in range(len(breakpoints) - 1):
        b1, b2 = breakpoints[i], breakpoints[i + 1]

        if b2 - b1 == 1:
            bin_ids[b1:b2] = offset
            offset += 1
        else:
            this_bin_ids = greedy_binning_nobreak(
                block_lengths[b1:b2],
                block_umi[b1:b2],
                block_snp_umi[b1:b2],
                block_normal_umi[b1:b2],
                secondary_min_umi,
                secondary_min_snp_umi,
                secondary_min_normal_umi,
                max_binlength,
            )

            bin_ids[b1:b2] = offset + this_bin_ids
            offset += np.max(this_bin_ids) + 1

    if "bin_id" in df_gene_snp.columns:
        logger.warning(f"Overwriting bin_id column, storing in block_id.")
        df_gene_snp["block_id"] = df_gene_snp["bin_id"]

    # NB map each block_id to its bin id
    df_gene_snp["bin_id"] = getattr(df_gene_snp, key).map(
        {i: x for i, x in enumerate(bin_ids)}
    )

    # NB return df_gene_snp with an updated bin_id column, and potentially deifned block_id column if it was overwritten.
    return df_gene_snp


# @cacher("binned_counts.hdf5")
def summarize_counts_for_bins(
    df_gene_snp,
    adata,
    single_X,
    single_total_bb_RD,
    phase_indicator,
    nu,
    logphase_shift,
    geneticmap_file,
):
    """
    Attributes:
    ----------
    df_gene_snp : pd.DataFrame
        Contain "block_id" column to indicate which genes/snps belong to which block.

    Returns
    ----------
    lengths : array, (n_chromosomes,)
        Number of blocks per chromosome.

    single_X : array, (n_bins, 2, n_spots)
        Transcript counts and B allele count per bin per cell.

    single_base_nb_mean : array, (n_bins, n_spots)
        Baseline transcript counts in normal diploid per bin per cell.

    single_total_bb_RD : array, (n_bins, n_spots)
        Total allele count per bin per cell.

    log_sitewise_transmat : array, (n_bins,)
        Log phase switch probability between each pair of adjacent bins.
    """
    logger.info(f"Summarizing counts for bins.")

    has_assigned_bin = ~df_gene_snp.bin_id.isnull()

    bins = df_gene_snp.loc[has_assigned_bin, "bin_id"].unique()

    # NB last axis is the number of spot (barcodes).
    n_bins = len(bins)
    n_spots = adata.shape[0]

    bin_single_X = np.zeros((n_bins, 2, n_spots), dtype=int)
    bin_single_base_nb_mean = np.zeros((n_bins, n_spots))
    bin_single_total_bb_RD = np.zeros((n_bins, n_spots), dtype=int)

    logger.info(
        f"Retaining {100. * np.mean(has_assigned_bin):.2f}% of gene/snps with assigned bin."
    )

    # NB unique block ids and gene names per bin.
    df_bin_contents = (
        df_gene_snp[has_assigned_bin]
        .groupby("bin_id", sort=True)
        .agg({"block_id": set, "gene": set})
    )

    if df_bin_contents.index.isna().any():
        logger.warning(f"Found ill-defined group with None entries for group.")

    block_sets = df_bin_contents["block_id"].to_numpy()
    gene_sets = df_bin_contents["gene"].to_numpy()

    gene_names = adata.var.index.to_numpy()
    gene_index_map = {g: i for i, g in enumerate(gene_names)}
    count_matrix = adata.layers["count"]  # (n_spots, n_genes), sparse or dense.

    for b in range(df_bin_contents.shape[0]):
        # BAF (SNPs): gather involved blocks
        involved_blocks = [x for x in block_sets[b] if x is not None]
        if involved_blocks:
            ib = np.fromiter(involved_blocks, dtype=int)
            # phased B counts per block
            phased = np.where(
                phase_indicator[ib].reshape(-1, 1),
                single_X[ib, 1, :],
                single_total_bb_RD[ib, :] - single_X[ib, 1, :],
            )
            # NB H0 counts for each bin (summed over blocks).
            bin_single_X[b, 1, :] = phased.sum(axis=0)

            # NB H0+H1 counts for each bin (summed over blocks).
            bin_single_total_bb_RD[b, :] = single_total_bb_RD[ib, :].sum(axis=0)

        # RDR (genes): gather involved gene indices
        involved_genes = [x for x in gene_sets[b] if x is not None]
        if involved_genes:
            gene_idx = [
                gene_index_map[g] for g in involved_genes if g in gene_index_map
            ]
            if gene_idx:
                block_sum = count_matrix[:, gene_idx].sum(axis=1)
                # Handle scipy.sparse result
                bin_single_X[b, 0, :] = np.asarray(block_sum).ravel()
        else:
            logger.debug(f"No genes found for bin row {b}.")

    chr_order = df_gene_snp.CHR.unique()
    lengths = (
        df_gene_snp.loc[has_assigned_bin]
        .groupby("CHR")["bin_id"]
        .nunique()
        .reindex(chr_order, fill_value=0)
        .to_numpy()
    )

    assert bin_single_X.ndim == 3

    return SpatioGenomicCounts(
        lengths, bin_single_X, bin_single_base_nb_mean, bin_single_total_bb_RD
    )
