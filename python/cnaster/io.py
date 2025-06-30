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
    
def load_joint_data(
    input_filelist,
    snp_dir,
    alignment_files,
    filtergenelist_file,
    filterregion_file,
    normalidx_file,
    min_snpumis=50,
    min_percent_expressed_spots=0.005,
    local_outlier_filter=True,
):
    ##### read meta sample info #####
    df_meta = pd.read_csv(input_filelist, sep="\t", header=None)
    df_meta.rename(
        columns=dict(zip(df_meta.columns[:3], ["bam", "sample_id", "spaceranger_dir"])),
        inplace=True,
    )
    
    logger.info(f"Input spaceranger file list {input_filelist} contains:")
    logger.info(df_meta)
    
    df_barcode = pd.read_csv(
        f"{snp_dir}/barcodes.txt", header=None, names=["combined_barcode"]
    )
    df_barcode["sample_id"] = [
        x.split("_")[-1] for x in df_barcode.combined_barcode.values
    ]
    df_barcode["barcode"] = [
        x.split("_")[0] for x in df_barcode.combined_barcode.values
    ]
    
    ##### read SNP count #####
    cell_snp_Aallele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Aallele.npz")
    cell_snp_Ballele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Ballele.npz")
    
    unique_snp_ids = np.load(f"{snp_dir}/unique_snp_ids.npy", allow_pickle=True)
    snp_barcodes = pd.read_csv(
        f"{snp_dir}/barcodes.txt", header=None, names=["barcodes"]
    )

    assert (len(alignment_files) == 0) or (len(alignment_files) + 1 == df_meta.shape[0])

    ##### read anndata and coordinate #####
    # add position
    adata = None
    for i, sname in enumerate(df_meta.sample_id.values):
        # locate the corresponding rows in df_meta
        index = np.where(df_barcode["sample_id"] == sname)[0]
        df_this_barcode = copy.copy(df_barcode.iloc[index, :])
        df_this_barcode.index = df_this_barcode.barcode
        # read adata count info
        if Path(
            f"{df_meta['spaceranger_dir'].iloc[i]}/filtered_feature_bc_matrix.h5"
        ).exists():
            adatatmp = sc.read_10x_h5(
                f"{df_meta['spaceranger_dir'].iloc[i]}/filtered_feature_bc_matrix.h5"
            )
        elif Path(
            f"{df_meta['spaceranger_dir'].iloc[i]}/filtered_feature_bc_matrix.h5ad"
        ).exists():
            adatatmp = sc.read_h5ad(
                f"{df_meta['spaceranger_dir'].iloc[i]}/filtered_feature_bc_matrix.h5ad"
            )
        else:
            logging.error(
                f"{df_meta['spaceranger_dir'].iloc[i]} directory doesn't have a filtered_feature_bc_matrix.h5 or filtered_feature_bc_matrix.h5ad file!"
            )

        adatatmp.layers["count"] = adatatmp.X.A
        # reorder anndata spots to have the same order as df_this_barcode
        idx_argsort = pd.Categorical(
            adatatmp.obs.index, categories=list(df_this_barcode.barcode), ordered=True
        ).argsort()
        adatatmp = adatatmp[idx_argsort, :]
        # read position info
        if Path(
            f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions.csv"
        ).exists():
            df_this_pos = pd.read_csv(
                f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions.csv",
                sep=",",
                header=0,
                names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"],
            )
        elif Path(
            f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions_list.csv"
        ).exists():
            df_this_pos = pd.read_csv(
                f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions_list.csv",
                sep=",",
                header=None,
                names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"],
            )
        else:
            raise Exception("No spatial coordinate file!")
        df_this_pos = df_this_pos[df_this_pos.in_tissue == True]
        # only keep shared barcodes
        shared_barcodes = set(list(df_this_pos.barcode)) & set(list(adatatmp.obs.index))
        adatatmp = adatatmp[adatatmp.obs.index.isin(shared_barcodes), :]
        df_this_pos = df_this_pos[df_this_pos.barcode.isin(shared_barcodes)]
        #
        # df_this_pos.barcode = pd.Categorical(df_this_pos.barcode, categories=list(df_this_barcode.barcode), ordered=True)
        df_this_pos.barcode = pd.Categorical(
            df_this_pos.barcode, categories=list(adatatmp.obs.index), ordered=True
        )
        df_this_pos.sort_values(by="barcode", inplace=True)
        adatatmp.obsm["X_pos"] = np.vstack([df_this_pos.x, df_this_pos.y]).T
        adatatmp.obs["sample"] = sname
        adatatmp.obs.index = [f"{x}_{sname}" for x in adatatmp.obs.index]
        adatatmp.var_names_make_unique()
        if adata is None:
            adata = adatatmp
        else:
            adata = anndata.concat([adata, adatatmp], join="outer")
    # replace nan with 0
    adata.layers["count"][np.isnan(adata.layers["count"])] = 0
    adata.layers["count"] = adata.layers["count"].astype(int)

    # shared barcodes between adata and SNPs
    shared_barcodes = set(list(snp_barcodes.barcodes)) & set(list(adata.obs.index))
    cell_snp_Aallele = cell_snp_Aallele[snp_barcodes.barcodes.isin(shared_barcodes), :]
    cell_snp_Ballele = cell_snp_Ballele[snp_barcodes.barcodes.isin(shared_barcodes), :]
    snp_barcodes = snp_barcodes[snp_barcodes.barcodes.isin(shared_barcodes)]
    adata = adata[adata.obs.index.isin(shared_barcodes), :]
    adata = adata[
        pd.Categorical(
            adata.obs.index, categories=list(snp_barcodes.barcodes), ordered=True
        ).argsort(),
        :,
    ]

    ##### load pairwise alignments #####
    # TBD: directly convert to big "adjacency" matrix
    across_slice_adjacency_mat = None
    if len(alignment_files) > 0:
        EPS = 1e-6
        row_ind = []
        col_ind = []
        dat = []
        offset = 0
        for i, f in enumerate(alignment_files):
            pi = np.load(f)
            # normalize p such that max( rowsum(pi), colsum(pi) ) = 1, max alignment weight = 1
            pi = pi / np.max(np.append(np.sum(pi, axis=0), np.sum(pi, axis=1)))
            sname1 = df_meta.sample_id.values[i]
            sname2 = df_meta.sample_id.values[i + 1]
            assert pi.shape[0] == np.sum(
                df_barcode["sample_id"] == sname1
            )  # double check whether this is correct
            assert pi.shape[1] == np.sum(
                df_barcode["sample_id"] == sname2
            )  # or the dimension should be flipped
            # for each spot s in sname1, select {t: spot t in sname2 and pi[s,t] >= np.max(pi[s,:])} as the corresponding spot in the other slice
            for row in range(pi.shape[0]):
                cutoff = np.max(pi[row, :]) if np.max(pi[row, :]) > EPS else 1 + EPS
                list_cols = np.where(pi[row, :] >= cutoff - EPS)[0]
                row_ind += [offset + row] * len(list_cols)
                col_ind += list(offset + pi.shape[0] + list_cols)
                dat += list(pi[row, list_cols])
            offset += pi.shape[0]
        across_slice_adjacency_mat = scipy.sparse.csr_matrix(
            (dat, (row_ind, col_ind)), shape=(adata.shape[0], adata.shape[0])
        )
        across_slice_adjacency_mat += across_slice_adjacency_mat.T

    # filter out spots with too small number of UMIs
    indicator = np.sum(adata.layers["count"], axis=1) >= min_snpumis
    adata = adata[indicator, :]
    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]
    if not (across_slice_adjacency_mat is None):
        across_slice_adjacency_mat = across_slice_adjacency_mat[indicator, :][
            :, indicator
        ]

    # filter out spots with too small number of SNP-covering UMIs
    indicator = (
        np.sum(cell_snp_Aallele, axis=1).A.flatten()
        + np.sum(cell_snp_Ballele, axis=1).A.flatten()
        >= min_snpumis
    )
    adata = adata[indicator, :]
    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]
    if not (across_slice_adjacency_mat is None):
        across_slice_adjacency_mat = across_slice_adjacency_mat[indicator, :][
            :, indicator
        ]

    # filter out genes that are expressed in <min_percent_expressed_spots cells
    indicator = (
        np.sum(adata.X > 0, axis=0) >= min_percent_expressed_spots * adata.shape[0]
    ).A.flatten()
    genenames = set(list(adata.var.index[indicator]))
    adata = adata[:, indicator]
    print(adata)
    print(
        "median UMI after filtering out genes < 0.5% of cells = {}".format(
            np.median(np.sum(adata.layers["count"], axis=1))
        )
    )

    if not filtergenelist_file is None:
        filter_gene_list = get_filtergenelist(filtergenelist_file)
        
        indicator_filter = np.array(
            [(not x in filter_gene_list) for x in adata.var.index]
        )
        
        adata = adata[:, indicator_filter]
        
        logger.info(
            "median UMI after filtering out genes in filtergenelist_file = {}".format(
                np.median(np.sum(adata.layers["count"], axis=1))
            )
        )

    if not filterregion_file is None:
        ranges = ranges_get_filter_ranges(filter_range_file)
        
        indicator_filter = np.array([True] * cell_snp_Aallele.shape[1])
        j = 0
        
        for i in range(cell_snp_Aallele.shape[1]):
            this_chr = int(unique_snp_ids[i].split("_")[0])
            this_pos = int(unique_snp_ids[i].split("_")[1])

            # NB fast forward genomic position
            while j < ranges.shape[0] and (
                (ranges.Chr.values[j] < this_chr)
                or (
                    (ranges.Chr.values[j] == this_chr)
                    and (ranges.End.values[j] <= this_pos)
                )
            ):
                j += 1
            if (
                j < ranges.shape[0]
                and (ranges.Chr.values[j] == this_chr)
                and (ranges.Start.values[j] <= this_pos)
                and (ranges.End.values[j] > this_pos)
            ):
                indicator_filter[i] = False
                
        cell_snp_Aallele = cell_snp_Aallele[:, indicator_filter]
        cell_snp_Ballele = cell_snp_Ballele[:, indicator_filter]
        
        unique_snp_ids = unique_snp_ids[indicator_filter]

    if local_outlier_filter:
        clf = LocalOutlierFactor(n_neighbors=200)
        label = clf.fit_predict(np.sum(adata.layers["count"], axis=0).reshape(-1, 1))
        
        adata.layers["count"][:, np.where(label == -1)[0]] = 0
        
        logger.info("filter out {} outlier genes.".format(np.sum(label == -1)))

    if not normalidx_file is None:
        normal_barcodes = pd.read_csv(normalidx_file, header=None).iloc[:, 0].values
        adata.obs["tumor_annotation"] = "tumor"
        adata.obs["tumor_annotation"][adata.obs.index.isin(normal_barcodes)] = "normal"
        
        logger.info(adata.obs["tumor_annotation"].value_counts())

    return (
        adata,
        cell_snp_Aallele.A,
        cell_snp_Ballele.A,
        unique_snp_ids,
        across_slice_adjacency_mat,
    )
