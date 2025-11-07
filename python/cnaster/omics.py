import logging

import numpy as np
import pandas as pd
from cnaster.recomb import assign_centiMorgans, compute_numbat_phase_switch_prob
from cnaster.reference import get_reference_genes, get_reference_recomb_rates

logger = logging.getLogger(__name__)


# TODO assumes reference gene contains all those present in Visium anndata.
def form_gene_snp_table(
    unique_snp_ids, hgtable_file, adata, num_preceeding_rows=50  # MAGIC
):
    logger.info(f"Forming gene & snp meta data.")
    logger.info(f"Retrieving reference genes: {hgtable_file}")

    # NB read gene info and keep only chr1-chr22 and genes appearing in adata
    df_hgtable = get_reference_genes(hgtable_file)

    logger.info(f"Filtering reference genes to those in visium: {hgtable_file}")

    common_genes = set(df_hgtable.name2) & set(adata.var.index)
    genes_not_in_reference = set(adata.var.index) - common_genes

    logger.info(
        f"Found {100. * len(common_genes) / len(adata.var.index):.2f}% of visium genes to be in reference."
    )

    # TODO
    logger.info(f"Visium genes not in reference:")

    for gene in sorted(genes_not_in_reference):
        logger.info(gene)

    # NB limits reference genes to those present in (filtered) AnnData UMIs.
    df_hgtable = df_hgtable[df_hgtable.name2.isin(adata.var.index)]

    # NB a data frame including both gene and SNP info: CHR, START, END, snp_id, gene, is_interval
    df_gene = pd.DataFrame(
        {
            "CHR": [int(x[3:]) for x in df_hgtable.chrom.to_numpy()],
            "START": df_hgtable.cdsStart.to_numpy(),
            "END": df_hgtable.cdsEnd.to_numpy(),
            "snp_id": None,
            "gene": df_hgtable.name2.to_numpy(),
            "is_interval": True,
        }
    )

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

    # NB loops over SNPs.
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
        f"Retaining {100.0 * np.mean(isin[~df_gene_snp.is_interval]):.3f}% of SNPs with known gene (given Gencode filtered by AnnData) for num_preceeding_rows={num_preceeding_rows}."
    )

    logger.info(
        f"Failed to find overlapping gene for:\n{df_gene_snp[df_gene_snp.gene.isnull()]}"
    )

    df_gene_snp = df_gene_snp[isin]

    logger.info(f"Created gene-SNP table:\n{df_gene_snp.head()}")

    return df_gene_snp


def summarize_blocks(
    gene_snp_table,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
    block_key=None,
    normal_candidates=None
):
    assert block_key is not None, "block_key must be specified"
    assert block_key in gene_snp_table.columns, f"{block_key} not in DataFrame"

    map_snp_index = {x: i for i, x in enumerate(unique_snp_ids)}

    block_summary = gene_snp_table.groupby(block_key).agg(
        num_snps=("snp_id", lambda x: x.notna().sum()),
        num_genes=("is_interval", "sum"),
        genes=("gene", lambda x: list({g for g in x if g is not None})),
        snp_ids=("snp_id", lambda x: [s for s in x if s is not None]),
    )

    gene_names = adata.var.index.to_numpy()
    gene_index_map = {g: i for i, g in enumerate(gene_names)}
    count_matrix = adata.layers["count"]  # (n_spots, n_genes)

    total_umis = np.zeros(len(block_summary), dtype=int)
    snp_umis = np.zeros(len(block_summary), dtype=int)

    normal_umis = np.zeros(len(block_summary), dtype=int)
    normal_snp_umis = np.zeros(len(block_summary), dtype=int)

    if normal_candidates is not None:
        assert count_matrix.shape[0] == len(normal_candidates), f"{count_matrix.shape[0]} != {len(normal_candidates)}"
        
        assert cell_snp_Aallele.shape[0] == len(normal_candidates), f"{cell_snp_Aallele.shape[0]} != {len(normal_candidates)}"
        assert cell_snp_Ballele.shape[0] == len(normal_candidates), f"{cell_snp_Ballele.shape[0]} != {len(normal_candidates)}"
        
    for idx, (block_id, row) in enumerate(block_summary.iterrows()):
        genes = row["genes"]
        if genes:
            gene_idx = [gene_index_map[g] for g in genes if g in gene_index_map]
            if gene_idx:
                block_sum = count_matrix[:, gene_idx].sum()
                total_umis[idx] = int(block_sum)

                # Calculate normal spot UMIs
                if normal_candidates is not None:
                    normal_umis[idx] = int(count_matrix[normal_candidates, :][:, gene_idx].sum())

        # SNP-covering UMIs
        snp_ids = row["snp_ids"]
        if snp_ids:
            snp_idx = np.array([map_snp_index[s] for s in snp_ids])
            if len(snp_idx) > 0:
                snp_umis[idx] = int(
                    cell_snp_Aallele[:, snp_idx].sum()
                    + cell_snp_Ballele[:, snp_idx].sum()
                )

                # Calculate SNP-covering UMIs for normal spots
                if normal_candidates is not None:
                    normal_snp_umis[idx] = int(
                        cell_snp_Aallele[np.ix_(normal_candidates, snp_idx)].sum() +
                        cell_snp_Ballele[np.ix_(normal_candidates, snp_idx)].sum()
                    )
                    

    block_summary["total_umi"] = total_umis
    block_summary["snp_umi"] = snp_umis

    block_summary["normal_umi"] = normal_umis
    block_summary["normal_snp_umi"] = normal_snp_umis

    block_summary = block_summary.sort_values("total_umi", ascending=False)    
    
    logger.info(f"Breakdown of genes/SNPs/UMI per {block_key}:")
    logger.info(
        f"{'Block ID':<10}\t{'SNPs':>8}\t{'Genes':>8}\t{'Total UMI':>12}\t{'SNP UMI':>12}\t{'Normal UMI':>12}\t{'Normal SNP UMI':>12}"
    )
    logger.info("-" * 100)

    for block_id, row in block_summary.iterrows():
        logger.info(
            f"{block_id:<10}\t{row['num_snps']:>8}\t{row['num_genes']:>8}\t"
            f"{row['total_umi']:>12}\t{row['snp_umi']:>12}\t{row['normal_umi']:>12}\t{row['normal_snp_umi']:>12}"
        )

    # Summary statistics
    logger.info(
        f"\n"
        f"median snps/block: {block_summary['num_snps'].median():.1f},\n"
        f"median genes/block: {block_summary['num_genes'].median():.1f},\n"
        f"median umis/block: {block_summary['total_umi'].median():.1f},\n"
        f"median snp-umis/block: {block_summary['snp_umi'].median():.1f},\n"
        f"total blocks: {len(block_summary)},\n"
        f"total umis: {block_summary['total_umi'].sum()},\n"
        f"total snp-umis: {block_summary['snp_umi'].sum()},\n"
        f"total normal umis: {block_summary['normal_umi'].sum()},\n"
        f"total normal snp-umis: {block_summary['normal_snp_umi'].sum()}\n"
    )


def assign_initial_blocks(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
    initial_min_umi,
):
    """
    Initially assigns SNPs to blocks along the genome, based on merging overlapping gene intervals
    and requiring blocks have a minimm number of SNP covering reads.

    Returns
    ----------
    df_gene_snp : data frame, names: (CHR, START, END, snp_id, gene, is_interval, block_id)
         Gene and SNP info combined into a single dataframe sorted by (CHR, START).
        "is_interval"=True is a gene, otherwise SNP.
        "gene" contains the name of a gene, or the gene a SNP belongs.
    """
    logger.info(f"Assigning initial blocks")

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
        # NB overlap of df_gene_snp with block_genome_interval.
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

    assert np.all(df_gene_snp["initial_block_id"].values) >= 0, "TODO!"

    logger.info(
        "Assigned SNPs to initial blocks (intervals formed by overlapping genes)."
    )

    summarize_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        block_key="initial_block_id",
    )

    # NB second level: group the first level blocks into "haplotype blocks" such that the minimum SNP-covering UMI counts >= initial_min_umi.
    #    maps snp id, {chr}_{pos}_{ref}_{alt} to integer index.
    map_snp_index = {x: i for i, x in enumerate(unique_snp_ids)}
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
                    f"Reached last block with {this_snp_umis}/{initial_min_umi} required SNP UMIs."
                )
                break

            if change_chr:
                t -= 1

                # re-count SNP-covering UMIs
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
                    f"Reached contig end with {this_snp_umis}/{initial_min_umi} required SNP UMIs."
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

    # NB record the block id in df_gene_snps
    df_gene_snp["block_id"] = 0

    for i, x in enumerate(block_ranges_new):
        df_gene_snp.iloc[x[0] : x[1], -1] = i

    logger.info(
        f"Updating block assignment based on input phased genotypes and min. snp-covering UMI threshold={initial_min_umi}"
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


def summarize_counts_for_blocks_legacy(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
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
    # NB block_ids formed by merging overlapping genes into intervals, merging said intervals
    #    until a threshold min. snp-covering reads and assigning counts to intervals below.
    blocks = df_gene_snp.block_id.unique()

    # NB (num. intervals, 2, num. spots).
    single_X = np.zeros((len(blocks), 2, adata.shape[0]), dtype=int)

    single_base_nb_mean = np.zeros((len(blocks), adata.shape[0]))
    single_total_bb_RD = np.zeros((len(blocks), adata.shape[0]), dtype=int)

    # NB summarize counts of involved genes and SNPs for each block.
    map_snp_index = {x: i for i, x in enumerate(unique_snp_ids)}

    df_block_contents = df_gene_snp.groupby("block_id").agg(
        {"snp_id": list, "gene": list}
    )

    logger.info(f"Summarizing counts for blocks")

    # NB loop over blocks.
    for b in range(df_block_contents.shape[0]):
        logger.info(f"Solved for block {b}/{df_block_contents.shape[0]}")

        # NB BAF (SNPs)
        involved_snps_ids = [
            x for x in df_block_contents.snp_id.to_numpy()[b] if x is not None
        ]

        involved_snp_idx = np.array([map_snp_index[x] for x in involved_snps_ids])

        if len(involved_snp_idx) > 0:
            # NB sum haplotype A counts for SNPs in block.
            single_X[b, 1, :] = np.sum(cell_snp_Aallele[:, involved_snp_idx], axis=1)

            # NB sum haplotype A + haplotype B counts for SNPs in block.
            single_total_bb_RD[b, :] = np.sum(
                cell_snp_Aallele[:, involved_snp_idx], axis=1
            ) + np.sum(cell_snp_Ballele[:, involved_snp_idx], axis=1)

        # RDR (genes)
        involved_genes = list(
            set([x for x in df_block_contents.gene.to_numpy()[b] if x is not None])
        )

        if len(involved_genes) > 0:
            # NB sum of umis for all genes in block.
            single_X[b, 0, :] = np.sum(
                adata.layers["count"][:, adata.var.index.isin(involved_genes)], axis=1
            )
        else:
            logger.warning(f"No genes found for block {b}.")

    # NB array of number of unique blocks by contig.
    lengths = np.zeros(len(df_gene_snp.CHR.unique()), dtype=int)

    for i, c in enumerate(df_gene_snp.CHR.unique()):
        lengths[i] = len(df_gene_snp[df_gene_snp.CHR == c].block_id.unique())

    assert single_X.ndim == 3

    # NB single_base_nb_mean is currently all zeros.
    return (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
    )


def summarize_counts_for_blocks(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
):
    logger.info(f"Summarizing counts for blocks")

    # precompute mapping: snp_id -> index
    map_snp_index = {x: i for i, x in enumerate(unique_snp_ids)}

    # filter to SNPs only (drop genes)
    df_snps = df_gene_snp[df_gene_snp.snp_id.notna()].copy()
    df_snps["snp_idx"] = df_snps.snp_id.map(map_snp_index)

    # group SNPs by block_id and aggregate indices as lists
    snp_groups = df_snps.groupby("block_id")["snp_idx"].apply(np.array)

    # TODO HACK?  df_gene_snp.gene.notna()
    df_genes = df_gene_snp[df_gene_snp.is_interval == True].copy()
    gene_groups = df_genes.groupby("block_id")["gene"].apply(lambda x: list(set(x)))

    # NB block_ids formed by merging overlapping genes into intervals, merging said intervals
    #    until a threshold min. snp-covering reads and assigning counts to intervals below.
    blocks = df_gene_snp.block_id.unique()
    n_blocks = len(blocks)
    n_spots = adata.shape[0]

    single_X = np.zeros((n_blocks, 2, n_spots), dtype=int)
    single_base_nb_mean = np.zeros((n_blocks, n_spots))
    single_total_bb_RD = np.zeros((n_blocks, n_spots), dtype=int)

    # precompute gene counts if using sparse matrix (for efficiency)
    gene_counts = adata.layers["count"]  # (n_spots, n_genes)
    gene_names = adata.var.index.to_numpy()

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
            gene_mask = np.isin(gene_names, genes)
            if gene_mask.any():
                single_X[block_id, 0, :] = gene_counts[:, gene_mask].sum(axis=1)

    lengths = df_gene_snp.groupby("CHR")["block_id"].nunique().to_numpy()

    assert single_X.ndim == 3

    return (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
    )


def get_sitewise_transmat(df_gene_snp, geneticmap_file, nu, logphase_shift):
    """
    Phase switch probability from recombination rate / genetic distance [cM].
    """
    logger.info(
        f"Constructing sitewise transition matrix for phasing given recombination rates."
    )

    # NB define recombination rates.
    ref_positions_cM = get_reference_recomb_rates(geneticmap_file)

    # NB sorted contig,start per block.
    sorted_chr_pos_first = df_gene_snp.groupby("block_id").agg(
        {"CHR": "first", "START": "first"}
    )

    # NB dataframe to list.
    sorted_chr_pos_first = list(
        zip(sorted_chr_pos_first.CHR.to_numpy(), sorted_chr_pos_first.START.to_numpy())
    )

    sorted_chr_pos_last = df_gene_snp.groupby("block_id").agg(
        {"CHR": "last", "END": "last"}
    )

    sorted_chr_pos_last = list(
        zip(sorted_chr_pos_last.CHR.to_numpy(), sorted_chr_pos_last.END.to_numpy())
    )

    # NB [(chr1, start1), (chr1, end1), (chr2, start2), (chr2, end2), ...]) construct ...
    tmp_sorted_chr_pos = [
        val for pair in zip(sorted_chr_pos_first, sorted_chr_pos_last) for val in pair
    ]

    # NB positions in cM of [(chr1, start1), (chr1, end1), (chr2, start2), (chr2, end2), ...])
    position_cM = assign_centiMorgans(tmp_sorted_chr_pos, ref_positions_cM)

    # NB tmp_sorted_chr_pos used to identify chromosome switches.
    phase_switch_prob = compute_numbat_phase_switch_prob(
        position_cM, tmp_sorted_chr_pos, nu
    )

    # NB transition matrix for phasing.
    log_sitewise_transmat = np.minimum(
        np.log(0.5), np.log(phase_switch_prob) - logphase_shift
    )

    # NB positions -> pairs by sampling at rate 2.
    log_sitewise_transmat = log_sitewise_transmat[
        np.arange(1, len(log_sitewise_transmat), 2)
    ]

    # NB returns array.
    return log_sitewise_transmat


def greedy_binning_nobreak_legacy(block_lengths, block_umi, secondary_min_umi, max_binlength):
    """
    Given a set of blocks, find new blocks that meet a requirement on the minimum number
    of UMIs and do not exceed max_binlength.
    """
    assert len(block_lengths) == len(block_umi)

    bin_ranges = []
    s = 0

    while s < len(block_lengths):
        t = s + 1

        # NB extend included blocks until meets required umi count.
        while t < len(block_lengths) and np.sum(block_umi[s:t]) < secondary_min_umi:
            t += 1

            # NB current block is too long, time to split & meets SNP UMI count.
            if (np.sum(block_lengths[s:t]) >= max_binlength) and (
                np.sum(block_umi[s:t]) >= secondary_min_umi
            ):
                logger.warning(
                    f"Solved for block with length={np.sum(block_lengths[s:t])/max_binlength} [max_binlength] given secondary_min_umi threshold."
                )

                t = max(t - 1, s + 1)
                break

        # NB check whether it is a very small bin at the end.
        if (
            s > 0
            and t == len(block_lengths)
            and np.sum(block_umi[s:t]) < secondary_min_umi
            # TODO HACK
            # and np.sum(block_umi[s:t]) < 0.5 * secondary_min_umi
            # and np.sum(block_lengths[s:t]) < 0.5 * max_binlength
        ):
            logger.debug(
                f"Last block failed secondary_min_umi filter with fraction={np.sum(block_umi[s:t]) / secondary_min_umi:.3f}, merging with previous."
            )
            bin_ranges[-1][1] = t
        else:
            bin_ranges.append([s, t])

        s = t

    bin_ids = np.zeros(len(block_lengths), dtype=int)

    for i, x in enumerate(bin_ranges):
        bin_ids[x[0] : x[1]] = i

    return bin_ids


def create_bin_ranges_legacy(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
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

        NB should contain a phase column, derived from pop. + copy based phasing.

    single_total_bb_RD : array, (n_blocks, n_spots)
        Total SNP-covering reads per haplotype block per spot.

    refined_lengths : array
        Number of haplotype blocks before each phase switch. The numbers should sum up to n_blocks.

    Returns
    -------
    df_gene_snp : data frame, (CHR, START, END, snp_id, gene, is_interval, block_id, bin_id)
        The newly added bin_id column indicates which bin each gene or SNP belongs to.
    """
    logger.info(f"Recalculating blocks given new phasing.")

    # NB block intervals, sorted by contig and start?
    sorted_chr_pos_both = df_gene_snp.groupby("block_id").agg(
        {"CHR": "first", "START": "first", "END": "last"}
    )

    block_lengths = (
        sorted_chr_pos_both.END.to_numpy() - sorted_chr_pos_both.START.to_numpy()
    )
    n_blocks = len(block_lengths)

    # NB summed across spots.
    block_umi = np.sum(single_total_bb_RD, axis=1)

    logger.info(
        f"Creating bin ranges assuming a max length of {max_binlength} and min. block UMI of {secondary_min_umi}."
    )

    # TODO max_binlength.
    # NB get a list of points where existing block must be broken as too long.
    #    refined_lengths derived (tangentially) from phasing - represents contig boundaries; forced break when minor BAF changes by e.g. 0.1
    breakpoints = np.concatenate(
        [
            np.cumsum(refined_lengths),
            np.where(block_lengths > max_binlength)[0],
            np.where(block_lengths > max_binlength)[0] + 1,
        ]
    )

    breakpoints = np.sort(np.unique(breakpoints))

    # NB append 0 in the front of breakpoints so that each pair of adjacent
    #    breakpoints can be an input to greedy_binning_nobreak; occurs if
    #    block_lengths[0] < max_binlength.
    if breakpoints[0] != 0:
        breakpoints = np.append([0], breakpoints)

    assert np.all(breakpoints[:-1] < breakpoints[1:])

    # NB loop over breakpoints and bin each block
    bin_ids = np.zeros(n_blocks, dtype=int)

    # NB cumulative count of assigned bin ids.
    offset = 0

    for i in range(len(breakpoints) - 1):
        b1, b2 = breakpoints[i], breakpoints[i + 1]

        if b2 - b1 == 1:
            bin_ids[b1:b2] = offset
            offset += 1
        else:
            this_bin_ids = greedy_binning_nobreak(
                block_lengths[b1:b2], block_umi[b1:b2], secondary_min_umi, max_binlength
            )
            bin_ids[b1:b2] = offset + this_bin_ids
            offset += np.max(this_bin_ids) + 1

    # NB append bin_ids to df_gene_snp
    df_gene_snp["bin_id"] = df_gene_snp.block_id.map(
        {i: x for i, x in enumerate(bin_ids)}
    )

    summarize_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        block_key="bin_id",
    )
    
    return df_gene_snp


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
    assert len(block_lengths) == len(block_umi) == len(block_snp_umi) == len(block_normal_umi)

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

            # Check if all min requirements are met
            meets_umi = total_umi >= secondary_min_umi
            meets_snp = snp_umi >= secondary_min_snp_umi
            meets_normal = normal_umi >= secondary_min_normal_umi
            all_criteria_met = meets_umi and meets_snp and meets_normal

            # Break if bin is too long but meets UMI requirements
            if length >= max_binlength and all_criteria_met:
                logger.warning(
                    f"Solved for bin with length={length/max_binlength:.2f} [max_binlength] "
                    f"(UMI={total_umi}, SNP-UMI={snp_umi}, normal-UMI={normal_umi})"
                )
                t = max(t - 1, s + 1)
                break

            # Continue if criteria not met and not too long
            if all_criteria_met:
                break

            t += 1

        # Final counts for bin [s:t]
        total_umi = np.sum(block_umi[s:t])
        snp_umi = np.sum(block_snp_umi[s:t])
        normal_umi = np.sum(block_normal_umi[s:t])
        length = np.sum(block_lengths[s:t])

        # Check if it's a small bin at the end that doesn't meet criteria
        if s > 0 and t == len(block_lengths):
            if (total_umi < secondary_min_umi or
                snp_umi < secondary_min_snp_umi or
                normal_umi < secondary_min_normal_umi):
                logger.debug(
                    f"Last bin failed thresholds "
                    f"(UMI={total_umi}/{secondary_min_umi}, "
                    f"SNP-UMI={snp_umi}/{secondary_min_snp_umi}, "
                    f"normal-UMI={normal_umi}/{secondary_min_normal_umi}), "
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

    return bin_ids


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
):
    """
    Aggregate haplotype blocks to bins with multiple UMI constraints.

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
    logger.info(f"Recalculating blocks given new phasing.")

    # Block intervals
    sorted_chr_pos_both = df_gene_snp.groupby("block_id").agg(
        {"CHR": "first", "START": "first", "END": "last"}
    )

    block_lengths = (
        sorted_chr_pos_both.END.to_numpy() - sorted_chr_pos_both.START.to_numpy()
    )
    n_blocks = len(block_lengths)

    # Total UMI per block (summed across spots)
    block_umi = np.sum(single_X[:, 0, :], axis=1)  # transcript counts

    # SNP-covering UMI per block
    block_snp_umi = np.sum(single_total_bb_RD, axis=1)

    # Normal-spot UMI per block
    if normal_candidates is not None:
        if isinstance(normal_candidates, (np.ndarray, pd.Series)) and normal_candidates.dtype == bool:
            normal_idx = np.flatnonzero(normal_candidates)
        else:
            normal_idx = np.asarray(normal_candidates, dtype=int)
            
        block_normal_umi = np.sum(single_X[:, 0, normal_idx], axis=1)
    else:
        block_normal_umi = np.zeros(n_blocks, dtype=int)
        secondary_min_normal_umi = 0  # disable normal constraint

    logger.info(
        f"Creating bin ranges: max_length={max_binlength}, "
        f"min_umi={secondary_min_umi}, "
        f"min_snp_umi={secondary_min_snp_umi}, "
        f"min_normal_umi={secondary_min_normal_umi}"
    )

    # Breakpoints from phase switches and oversized blocks
    breakpoints = np.concatenate(
        [
            np.cumsum(refined_lengths),
            np.where(block_lengths > max_binlength)[0],
            np.where(block_lengths > max_binlength)[0] + 1,
        ]
    )

    breakpoints = np.sort(np.unique(breakpoints))

    if breakpoints[0] != 0:
        breakpoints = np.append([0], breakpoints)

    assert np.all(breakpoints[:-1] < breakpoints[1:])

    # Assign bin IDs
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

    # Append bin_ids to df_gene_snp
    df_gene_snp["bin_id"] = df_gene_snp.block_id.map(
        {i: x for i, x in enumerate(bin_ids)}
    )

    summarize_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        block_key="bin_id",
        normal_candidates=normal_candidates,
    )

    return df_gene_snp


# TODO duplicates summarize_counts_for_blocks?
def summarize_counts_for_bins_legacy(
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
    logger.info(f"Summarizing counts for bins.")

    bins = df_gene_snp.bin_id.unique()

    # NB last axis is the number of spot (barcodes).
    bin_single_X = np.zeros((len(bins), 2, adata.shape[0]), dtype=int)

    bin_single_base_nb_mean = np.zeros((len(bins), adata.shape[0]))
    bin_single_total_bb_RD = np.zeros((len(bins), adata.shape[0]), dtype=int)

    has_assigned_bin = ~df_gene_snp.bin_id.isnull()

    logger.info(
        f"Retaining {100. * np.mean(has_assigned_bin)} of bins with assigned block."
    )

    # NB summarize counts of involved genes and blocks within each bin.
    df_bin_contents = (
        df_gene_snp[has_assigned_bin]
        .groupby("bin_id")
        .agg({"block_id": set, "gene": set})
    )

    # NB loop over bins (phased blocks meeting max. length and min. UMI requirements).
    for b in range(df_bin_contents.shape[0]):
        # BAF (SNPs)
        involved_blocks = [
            x for x in df_bin_contents.block_id.to_numpy()[b] if x is not None
        ]

        this_phased = np.where(
            phase_indicator[involved_blocks].reshape(-1, 1),
            single_X[involved_blocks, 1, :],
            single_total_bb_RD[involved_blocks, :] - single_X[involved_blocks, 1, :],
        )

        # NB H0 counts for each bin (summed over blocks).
        bin_single_X[b, 1, :] = np.sum(this_phased, axis=0)

        # NB H0+H1 counts for each bin (summed over blocks).
        bin_single_total_bb_RD[b, :] = np.sum(
            single_total_bb_RD[involved_blocks, :], axis=0
        )

        # RDR (genes)
        involved_genes = [
            x for x in df_bin_contents.gene.to_numpy()[b] if x is not None
        ]

        # NB all transcripts for genes in this bin.
        bin_single_X[b, 0, :] = np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)], axis=1
        )

    lengths = np.zeros(len(df_gene_snp.CHR.unique()), dtype=int)

    for i, c in enumerate(df_gene_snp.CHR.unique()):
        lengths[i] = len(
            df_gene_snp[
                (df_gene_snp.CHR == c) & (~df_gene_snp.bin_id.isnull())
            ].bin_id.unique()
        )

    # NB phase switch probability from genetic distance
    sorted_chr_pos_first = df_gene_snp.groupby("bin_id").agg(
        {"CHR": "first", "START": "first"}
    )

    sorted_chr_pos_first = list(
        zip(sorted_chr_pos_first.CHR.to_numpy(), sorted_chr_pos_first.START.to_numpy())
    )

    sorted_chr_pos_last = df_gene_snp.groupby("bin_id").agg(
        {"CHR": "last", "END": "last"}
    )

    sorted_chr_pos_last = list(
        zip(sorted_chr_pos_last.CHR.to_numpy(), sorted_chr_pos_last.END.to_numpy())
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

    assert bin_single_X.ndim == 3

    return (
        lengths,
        bin_single_X,
        bin_single_base_nb_mean,
        bin_single_total_bb_RD,
        log_sitewise_transmat,
    )


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
    # Use only assigned for shape; keeps original logic otherwise
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

    df_bin_contents = (
        df_gene_snp[has_assigned_bin]
        .groupby("bin_id", sort=True)
        .agg({"block_id": set, "gene": set})
    )
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

    # Array of number of unique bins by chromosome (vectorized)
    chr_order = df_gene_snp.CHR.unique()
    lengths = (
        df_gene_snp.loc[has_assigned_bin]
        .groupby("CHR")["bin_id"]
        .nunique()
        .reindex(chr_order, fill_value=0)
        .to_numpy()
    )

    # Phase switch probability from genetic distance (UNCHANGED)
    sorted_chr_pos_first = df_gene_snp.groupby("bin_id").agg(
        {"CHR": "first", "START": "first"}
    )
    sorted_chr_pos_first = list(
        zip(sorted_chr_pos_first.CHR.to_numpy(), sorted_chr_pos_first.START.to_numpy())
    )
    sorted_chr_pos_last = df_gene_snp.groupby("bin_id").agg(
        {"CHR": "last", "END": "last"}
    )
    sorted_chr_pos_last = list(
        zip(sorted_chr_pos_last.CHR.to_numpy(), sorted_chr_pos_last.END.to_numpy())
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

    assert bin_single_X.ndim == 3

    return (
        lengths,
        bin_single_X,
        bin_single_base_nb_mean,
        bin_single_total_bb_RD,
        log_sitewise_transmat,
    )
