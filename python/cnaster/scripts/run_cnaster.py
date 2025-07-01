from cnaster.io import load_sample_data
from cnaster.omics import form_gene_snp_table

(
    adata,
    cell_snp_Aallele.A,
    cell_snp_Ballele.A,
    unique_snp_ids,
    across_slice_adjacency_mat,
) = load_sample_data(
    spaceranger_meta_path,
    snp_dir,
    alignment_files,
    filter_gene_file,
    filter_range_file,
    normal_idx_file,
    min_snp_umis=50,
    min_percent_expressed_spots=5.0e-3,
    local_outlier_filter=True,
)

df_gene_snp = form_gene_snp_table(unique_snp_ids, config["hgtable_file"], adata)

# TODO assign initial fragment ranges based on over-lapping gene and min. snp covering umi count.
df_gene_snp = assign_initial_fragments(
    df_gene_snp, adata, cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids
)

(
    lengths,
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    log_sitewise_transmat,
) = summarize_counts_for_blocks(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
    nu=config["nu"],
    logphase_shift=config["logphase_shift"],
    geneticmap_file=config["geneticmap_file"],
)
