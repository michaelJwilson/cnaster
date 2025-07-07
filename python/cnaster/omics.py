import logging

import numpy as np
import pandas as pd

from cnaster.recomb import (assign_centiMorgans,
                            compute_numbat_phase_switch_prob)
from cnaster.reference import get_reference_genes, get_reference_recomb_rates

logger = logging.getLogger(__name__)


def form_gene_snp_table(unique_snp_ids, hgtable_file, adata):
    # NB read gene info and keep only chr1-chr22 and genes appearing in adata
    df_hgtable = get_reference_genes(hgtable_file)
    df_hgtable = df_hgtable[df_hgtable.name2.isin(adata.var.index)]

    # a data frame including both gene and SNP info: CHR, START, END, snp_id, gene, is_interval
    df_gene_snp = pd.DataFrame(
        {
            "CHR": [int(x[3:]) for x in df_hgtable.chrom.values],
            "START": df_hgtable.cdsStart.values,
            "END": df_hgtable.cdsEnd.values,
            "snp_id": None,
            "gene": df_hgtable.name2.values,
            "is_interval": True,
        }
    )

    # add SNP info
    snp_chr = np.array([int(x.split("_")[0]) for x in unique_snp_ids])
    snp_pos = np.array([int(x.split("_")[1]) for x in unique_snp_ids])

    df_gene_snp = pd.concat(
        [
            df_gene_snp,
            pd.DataFrame(
                {
                    "CHR": snp_chr,
                    "START": snp_pos,
                    "END": snp_pos + 1,
                    "snp_id": unique_snp_ids,
                    "gene": None,
                    "is_interval": False,
                }
            ),
        ],
        ignore_index=True,
    )

    df_gene_snp.sort_values(by=["CHR", "START"], inplace=True)

    # NB assign genes to each SNP:  for each SNP (with not null snp_id), find the previous gene (is_interval == True)
    #    such that the SNP start position is within the gene start & end interval.
    vec_is_interval = df_gene_snp.is_interval.values

    vec_chr = df_gene_snp.CHR.values
    vec_start = df_gene_snp.START.values
    vec_end = df_gene_snp.END.values

    for i in np.where(df_gene_snp.gene.isnull())[0]:
        # TODO first SNP has no gene.
        if i == 0:
            continue

        this_pos = vec_start[i]

        # NB decrement row indexes up to 50 behind (on same contig). 
        j = i - 1

        # TODO? overlapping genes: closest in start.
        while j >= 0 and j >= (i - 50) and (vec_chr[i] == vec_chr[j]):
            if (
                vec_is_interval[j]
                and vec_start[j] <= this_pos
                and vec_end[j] > this_pos
            ):
                df_gene_snp.iloc[i, 4] = df_gene_snp.iloc[j]["gene"]
                break

            j -= 1

    # NB remove SNPs that have no corresponding genes.
    isin = ~df_gene_snp.gene.isnull()

    logger.info(
        f"Retaining {100. * np.mean(isin[~df_gene_snp.is_interval]):.3f}% of SNPs according to known genes")
    )
    
    df_gene_snp = df_gene_snp[isin]

    return df_gene_snp


def assign_initial_blocks(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
    initial_min_umi=15,
):
    """
    Initially assigns SNPs to fragments along the genome.

    Returns
    ----------
    df_gene_snp : data frame, (CHR, START, END, snp_id, gene, is_interval, block_id)
        Gene and SNP info combined into a single data frame sorted by genomic positions.
        "is_interval" suggest whether the entry is a gene or a SNP.
        "gene" column either contain gene name if the entry is a gene, or the gene a SNP belongs to if the entry is a SNP.
    """
    # TODO un-necessary?
    # first level: partition of genome: by gene regions (if two genes overlap, they are grouped to one region)
    tmp_block_genome_intervals = list(
        zip(
            df_gene_snp[df_gene_snp.is_interval].CHR.values,
            df_gene_snp[df_gene_snp.is_interval].START.values,
            df_gene_snp[df_gene_snp.is_interval].END.values,
        )
    )

    block_genome_intervals = [tmp_block_genome_intervals[0]]
    merged = 0

    for x in tmp_block_genome_intervals[1:]:
        # check whether overlap with previous block
        if x[0] == block_genome_intervals[-1][0] and max(
            x[1], block_genome_intervals[-1][1]
        ) < min(x[2], block_genome_intervals[-1][2]):
            block_genome_intervals[-1] = (
                x[0],
                min(x[1], block_genome_intervals[-1][1]),
                max(x[2], block_genome_intervals[-1][2]),
            )

            # TODO warn on excessive length;
            merged += 1

        else:
            block_genome_intervals.append(x)

    logger.info(
        f"Merged {100. * merged / len(tmp_block_genome_intervals):.3f}% of input ranges;"
    )

    # get block_ranges in the index of df_gene_snp
    block_ranges = []

    for x in block_genome_intervals:
        indexes = np.where(
            (df_gene_snp.CHR.values == x[0])
            & (
                np.maximum(df_gene_snp.START.values, x[1])
                < np.minimum(df_gene_snp.END.values, x[2])
            )
        )[0]

        # index of rows into df_gene_snp that overlap each interval.
        # TODO can fail?
        block_ranges.append((indexes[0], indexes[-1] + 1))

    assert np.all(
        np.array([x[1] for x in block_ranges[:-1]])
        == np.array([x[0] for x in block_ranges[1:]])
    )

    # record the initial block id in df_gene_snps
    df_gene_snp["initial_block_id"] = 0

    for i, x in enumerate(block_ranges):
        df_gene_snp.iloc[x[0] : x[1], -1] = i

    # second level: group the first level blocks into "haplotype blocks" such that the minimum SNP-covering UMI counts >= initial_min_umi
    map_snp_index = {x: i for i, x in enumerate(unique_snp_ids)}
    initial_block_chr = df_gene_snp.CHR.values[np.array([x[0] for x in block_ranges])]
    block_ranges_new = []
    s = 0

    # TODO work through.
    while s < len(block_ranges):
        t = s

        while t <= len(block_ranges):
            t += 1

            reach_end = t == len(block_ranges)

            # TODO BUG? (not reach_end) and ...
            change_chr = initial_block_chr[s] != initial_block_chr[t - 1]

            # count SNP-covering UMI
            involved_snps_ids = df_gene_snp[
                (df_gene_snp.initial_block_id >= s) & (df_gene_snp.initial_block_id < t)
            ].snp_id

            involved_snps_ids = involved_snps_ids[~involved_snps_ids.isnull()].values
            involved_snp_idx = np.array([map_snp_index[x] for x in involved_snps_ids])

            this_snp_umis = (
                0
                if len(involved_snp_idx) == 0
                else np.sum(cell_snp_Aallele[:, involved_snp_idx])
                + np.sum(cell_snp_Ballele[:, involved_snp_idx])
            )

            if reach_end:
                break

            if change_chr:
                t -= 1

                # re-count SNP-covering UMIs
                involved_snps_ids = df_gene_snp.snp_id.iloc[
                    block_ranges[s][0] : block_ranges[t - 1][1]
                ]
                involved_snps_ids = involved_snps_ids[
                    ~involved_snps_ids.isnull()
                ].values

                involved_snp_idx = np.array(
                    [map_snp_index[x] for x in involved_snps_ids]
                )

                this_snp_umis = (
                    0
                    if len(involved_snp_idx) == 0
                    else np.sum(cell_snp_Aallele[:, involved_snp_idx])
                    + np.sum(cell_snp_Ballele[:, involved_snp_idx])
                )

                break

            if this_snp_umis >= initial_min_umi:
                break

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

        s = t

    # record the block id in df_gene_snps
    df_gene_snp["block_id"] = 0

    for i, x in enumerate(block_ranges_new):
        df_gene_snp.iloc[x[0] : x[1], -1] = i

    df_gene_snp = df_gene_snp.drop(columns=["initial_block_id"])

    return df_gene_snp


def summarize_counts_for_blocks(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
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

    single_X : array, (n_blocks, 2, n_spots)
        Transcript counts and B allele count per block per cell.

    single_base_nb_mean : array, (n_blocks, n_spots)
        Baseline transcript counts in normal diploid per block per cell.

    single_total_bb_RD : array, (n_blocks, n_spots)
        Total allele count per block per cell.

    log_sitewise_transmat : array, (n_blocks,)
        Log phase switch probability between each pair of adjacent blocks.
    """
    blocks = df_gene_snp.block_id.unique()

    single_X = np.zeros((len(blocks), 2, adata.shape[0]), dtype=int)

    single_base_nb_mean = np.zeros((len(blocks), adata.shape[0]))
    single_total_bb_RD = np.zeros((len(blocks), adata.shape[0]), dtype=int)

    # summarize counts of involved genes and SNPs within each block
    map_snp_index = {x: i for i, x in enumerate(unique_snp_ids)}

    df_block_contents = df_gene_snp.groupby("block_id").agg(
        {"snp_id": list, "gene": list}
    )

    # NB loop over "blocks"
    for b in range(df_block_contents.shape[0]):
        # BAF (SNPs)
        involved_snps_ids = [
            x for x in df_block_contents.snp_id.values[b] if x is not None
        ]

        involved_snp_idx = np.array([map_snp_index[x] for x in involved_snps_ids])

        if len(involved_snp_idx) > 0:
            # NB sum haplotype A allele across block: not A is defined by Eagle2 0/1 vs 1/0;
            single_X[b, 1, :] = np.sum(cell_snp_Aallele[:, involved_snp_idx], axis=1)

            # NB Eagle2 phased "REF" vs "ALT" counts.
            single_total_bb_RD[b, :] = np.sum(
                cell_snp_Aallele[:, involved_snp_idx], axis=1
            ) + np.sum(cell_snp_Ballele[:, involved_snp_idx], axis=1)

        # RDR (genes)
        involved_genes = list(
            set([x for x in df_block_contents.gene.values[b] if x is not None])
        )

        if len(involved_genes) > 0:
            # NB sum of umis for all genes in block.
            single_X[b, 0, :] = np.sum(
                adata.layers["count"][:, adata.var.index.isin(involved_genes)], axis=1
            )

    lengths = np.zeros(len(df_gene_snp.CHR.unique()), dtype=int)

    # NB lengths per block by chromosome.
    for i, c in enumerate(df_gene_snp.CHR.unique()):
        lengths[i] = len(df_gene_snp[df_gene_snp.CHR == c].block_id.unique())

    # NB -  phase switch probability from genetic distance.
    #    -  first chr and start of each block.
    sorted_chr_pos_first = df_gene_snp.groupby("block_id").agg(
        {"CHR": "first", "START": "first"}
    )

    # TODO
    sorted_chr_pos_first = list(
        zip(sorted_chr_pos_first.CHR.values, sorted_chr_pos_first.START.values)
    )

    sorted_chr_pos_last = df_gene_snp.groupby("block_id").agg(
        {"CHR": "last", "END": "last"}
    )
    sorted_chr_pos_last = list(
        zip(sorted_chr_pos_last.CHR.values, sorted_chr_pos_last.END.values)
    )

    tmp_sorted_chr_pos = [
        val for pair in zip(sorted_chr_pos_first, sorted_chr_pos_last) for val in pair
    ]

    ref_positions_cM = get_reference_recomb_rates(geneticmap_file)

    position_cM = assign_centiMorgans(tmp_sorted_chr_pos, ref_positions_cM)

    # NB tmp_sorted_chr_pos used to identify chromosome switches.
    phase_switch_prob = compute_numbat_phase_switch_prob(
        position_cM, tmp_sorted_chr_pos, nu
    )

    log_sitewise_transmat = np.minimum(
        np.log(0.5), np.log(phase_switch_prob) - logphase_shift
    )

    log_sitewise_transmat = log_sitewise_transmat[
        np.arange(1, len(log_sitewise_transmat), 2)
    ]

    return (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        log_sitewise_transmat,
    )


def create_bin_ranges(
    df_gene_snp,
    single_total_bb_RD,
    refined_lengths,
    secondary_min_umi,
    max_binlength=5e6,
):
    """
    Aggregate haplotype blocks to bins

    Attributes
    ----------
    df_gene_snp : data frame, (CHR, START, END, snp_id, gene, is_interval, block_id)
        Gene and SNP info combined into a single data frame sorted by genomic positions.
        "is_interval" suggest whether the entry is a gene or a SNP. "gene" column either
        contain gene name if the entry is a gene, or the gene a SNP belongs to if the entry is a SNP.

    single_total_bb_RD : array, (n_blocks, n_spots)
        Total SNP-covering reads per haplotype block per spot.

    refined_lengths : array
        Number of haplotype blocks before each phase switch. The numbers should sum up to n_blocks.

    Returns
    -------
    df_gene_snp : data frame, (CHR, START, END, snp_id, gene, is_interval, block_id, bin_id)
        The newly added bin_id column indicates which bin each gene or SNP belongs to.
    """

    def greedy_binning_nobreak(
        block_lengths, block_umi, secondary_min_umi, max_binlength
    ):
        """
        Returns
        -------
        bin_ids : array, (n_blocks)
            The bin id of the input blocks. Should have the same size with block_lengths and block_umi.
        """
        assert len(block_lengths) == len(block_umi)

        bin_ranges = []
        s = 0

        while s < len(block_lengths):
            t = s + 1

            while t < len(block_lengths) and np.sum(block_umi[s:t]) < secondary_min_umi:
                t += 1
                if np.sum(block_lengths[s:t]) >= max_binlength:
                    t = max(t - 1, s + 1)
                    break

            # check whether it is a very small bin in the end
            if (
                s > 0
                and t == len(block_lengths)
                and np.sum(block_umi[s:t]) < 0.5 * secondary_min_umi
                and np.sum(block_lengths[s:t]) < 0.5 * max_binlength
            ):
                bin_ranges[-1][1] = t
            else:
                bin_ranges.append([s, t])

            s = t

        bin_ids = np.zeros(len(block_lengths), dtype=int)

        for i, x in enumerate(bin_ranges):
            bin_ids[x[0] : x[1]] = i

        return bin_ids

    # block lengths and block umis
    sorted_chr_pos_both = df_gene_snp.groupby("block_id").agg(
        {"CHR": "first", "START": "first", "END": "last"}
    )

    block_lengths = sorted_chr_pos_both.END.values - sorted_chr_pos_both.START.values
    block_umi = np.sum(single_total_bb_RD, axis=1)

    n_blocks = len(block_lengths)

    # get a list of breakpoints where bin much break
    breakpoints = np.concatenate(
        [
            np.cumsum(refined_lengths),
            np.where(block_lengths > max_binlength)[0],
            np.where(block_lengths > max_binlength)[0] + 1,
        ]
    )

    breakpoints = np.sort(np.unique(breakpoints))

    # append 0 in the front of breakpoints so that each pair of adjacent breakpoints can be an input to greedy_binning_nobreak
    if breakpoints[0] != 0:
        breakpoints = np.append([0], breakpoints)

    assert np.all(breakpoints[:-1] < breakpoints[1:])

    # loop over breakpoints and bin each block
    bin_ids = np.zeros(n_blocks, dtype=int)
    offset = 0

    for i in range(len(breakpoints) - 1):
        b1 = breakpoints[i]
        b2 = breakpoints[i + 1]

        if b2 - b1 == 1:
            bin_ids[b1:b2] = offset
            offset += 1
        else:
            this_bin_ids = greedy_binning_nobreak(
                block_lengths[b1:b2], block_umi[b1:b2], secondary_min_umi, max_binlength
            )
            bin_ids[b1:b2] = offset + this_bin_ids
            offset += np.max(this_bin_ids) + 1

    # append bin_ids to df_gene_snp
    df_gene_snp["bin_id"] = df_gene_snp.block_id.map(
        {i: x for i, x in enumerate(bin_ids)}
    )

    return df_gene_snp


# TODO duplicates summarize_counts_for_blocks
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

    single_X : array, (n_blocks, 2, n_spots)
        Transcript counts and B allele count per block per cell.

    single_base_nb_mean : array, (n_blocks, n_spots)
        Baseline transcript counts in normal diploid per block per cell.

    single_total_bb_RD : array, (n_blocks, n_spots)
        Total allele count per block per cell.

    log_sitewise_transmat : array, (n_blocks,)
        Log phase switch probability between each pair of adjacent blocks.
    """
    bins = df_gene_snp.bin_id.unique()
    bin_single_X = np.zeros((len(bins), 2, adata.shape[0]), dtype=int)
    bin_single_base_nb_mean = np.zeros((len(bins), adata.shape[0]))
    bin_single_total_bb_RD = np.zeros((len(bins), adata.shape[0]), dtype=int)
    # summarize counts of involved genes and SNPs within each block
    df_bin_contents = (
        df_gene_snp[~df_gene_snp.bin_id.isnull()]
        .groupby("bin_id")
        .agg({"block_id": set, "gene": set})
    )
    for b in range(df_bin_contents.shape[0]):
        # BAF (SNPs)
        involved_blocks = [
            x for x in df_bin_contents.block_id.values[b] if x is not None
        ]
        this_phased = np.where(
            phase_indicator[involved_blocks].reshape(-1, 1),
            single_X[involved_blocks, 1, :],
            single_total_bb_RD[involved_blocks, :] - single_X[involved_blocks, 1, :],
        )
        bin_single_X[b, 1, :] = np.sum(this_phased, axis=0)
        bin_single_total_bb_RD[b, :] = np.sum(
            single_total_bb_RD[involved_blocks, :], axis=0
        )
        # RDR (genes)
        involved_genes = [x for x in df_bin_contents.gene.values[b] if x is not None]
        bin_single_X[b, 0, :] = np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)], axis=1
        )

    # lengths
    lengths = np.zeros(len(df_gene_snp.CHR.unique()), dtype=int)
    for i, c in enumerate(df_gene_snp.CHR.unique()):
        lengths[i] = len(
            df_gene_snp[
                (df_gene_snp.CHR == c) & (~df_gene_snp.bin_id.isnull())
            ].bin_id.unique()
        )

    # phase switch probability from genetic distance
    sorted_chr_pos_first = df_gene_snp.groupby("bin_id").agg(
        {"CHR": "first", "START": "first"}
    )
    sorted_chr_pos_first = list(
        zip(sorted_chr_pos_first.CHR.values, sorted_chr_pos_first.START.values)
    )
    sorted_chr_pos_last = df_gene_snp.groupby("bin_id").agg(
        {"CHR": "last", "END": "last"}
    )
    sorted_chr_pos_last = list(
        zip(sorted_chr_pos_last.CHR.values, sorted_chr_pos_last.END.values)
    )

    tmp_sorted_chr_pos = [
        val for pair in zip(sorted_chr_pos_first, sorted_chr_pos_last) for val in pair
    ]
    ref_positions_cM = get_reference_recomb_rates(geneticmap_file)

    position_cM = assign_centiMorgans(tmp_sorted_chr_pos, ref_positions_cM)

    phase_switch_prob = compute_numbat_phase_switch_prob(
        position_cM, tmp_sorted_chr_pos, nu
    )
    log_sitewise_transmat = np.minimum(
        np.log(0.5), np.log(phase_switch_prob) - logphase_shift
    )

    log_sitewise_transmat = log_sitewise_transmat[
        np.arange(1, len(log_sitewise_transmat), 2)
    ]

    return (
        lengths,
        bin_single_X,
        bin_single_base_nb_mean,
        bin_single_total_bb_RD,
        log_sitewise_transmat,
    )
