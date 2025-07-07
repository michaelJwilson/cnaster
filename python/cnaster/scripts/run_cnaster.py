import logging

from cnaster.config import YAMLConfig
# from cnaster.io import load_sample_data
#from cnaster.omics import (assign_initial_blocks, create_bin_ranges,
#                           form_gene_snp_table, summarize_counts_for_bins,
#                           summarize_counts_for_blocks)
#from cnaster.phasing import initial_phase_given_partition
#from cnaster.spatial import initialize_clones, multislice_adjacency

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def main():
    # HACK
    config_path = f"/u/mw9568/research/repos/cnaster/config_turing.yaml"
    config = YAMLConfig.from_file(config_path)

    """
    (
        adata,
        cell_snp_Aallele.A,
        cell_snp_Ballele.A,
        unique_snp_ids,
        across_slice_adjacency_mat,
    ) = load_sample_data(
        config
        filter_gene_file=config.references.filtergenelist_file,
        filter_range_file=config.references.filterregion_file,
    )

    df_gene_snp = form_gene_snp_table(unique_snp_ids, config["hgtable_file"], adata)

    # TODO assign initial fragment ranges based on over-lapping gene and min. snp covering umi count.
    df_gene_snp = assign_initial_blocks(
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

    initial_clone_for_phasing = initialize_clones(
        coords,
        sample_ids,
        x_part=config["npart_phasing"],
        y_part=config["npart_phasing"],
        single_tumor_prop=single_tumor_prop,
        threshold=config["tumorprop_threshold"],
    )

    phase_indicator, refined_lengths = initial_phase_given_partition(
        single_X,
        lengths,
        single_base_nb_mean,
        single_total_bb_RD,
        single_tumor_prop,
        initial_clone_for_phasing,
        5,
        log_sitewise_transmat,
        "sp",
        config["t_phaseing"],
        config["gmm_random_state"],
        config["fix_NB_dispersion"],
        config["shared_NB_dispersion"],
        config["fix_BB_dispersion"],
        config["shared_BB_dispersion"],
        30,
        1e-3,
        threshold=config["tumorprop_threshold"],
    )
    
    df_gene_snp["phase"] = np.where(
        df_gene_snp.snp_id.isnull(),
        None,
        df_gene_snp.block_id.map({i: x for i, x in enumerate(phase_indicator)}),
    )
    
    df_gene_snp = create_bin_ranges(
        df_gene_snp, single_total_bb_RD, refined_lengths, config["secondary_min_umi"]
    )
    
    (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        log_sitewise_transmat,
    ) = summarize_counts_for_bins(
        df_gene_snp,
        adata,
        single_X,
        single_total_bb_RD,
        phase_indicator,
        nu=config["nu"],
        logphase_shift=config["logphase_shift"],
        geneticmap_file=config["geneticmap_file"],
    )
    
    # expression count dataframe
    exp_counts = pd.DataFrame.sparse.from_spmatrix( scipy.sparse.csc_matrix(adata.layers["count"]), index=adata.obs.index, columns=adata.var.index)
    
    # smooth and adjacency matrix for each sample
    adjacency_mat, smooth_mat = multislice_adjacency(sample_ids, sample_list, coords, single_total_bb_RD, exp_counts, 
                                                     across_slice_adjacency_mat, construct_adjacency_method=config['construct_adjacency_method'], 
                                                     maxspots_pooling=config['maxspots_pooling'], construct_adjacency_w=config['construct_adjacency_w'])

    n_pooled = np.median(np.sum(smooth_mat > 0, axis=0).A.flatten())
    """
