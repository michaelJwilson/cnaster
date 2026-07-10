import numpy as np
import pandas as pd

from pathlib import Path
from cnaster.config import get_global_config, start_time
from cnaster.logger import get_logger
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix

logger = get_logger(__name__, start_time=start_time)


def load_clone_labels(single_X=None, config=None):
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

    if single_X is None:
        return initial_clone_index_baf, None

    known_rdr_normal = np.sum(single_X[:, 0, (clone_id == 0)], axis=-1)
    known_rdr_normal = known_rdr_normal / np.sum(known_rdr_normal)

    # TODO
    # bidx_inconfident = np.where(
    #     known_rdr_normal < config.quality.min_normal_count_perbin
    # )[0]
    # known_rdr_normal[bidx_inconfident] = 0

    spots_coverage = np.sum(single_X[:, 0, :], axis=0)

    known_single_base_nb_mean = known_rdr_normal.reshape(
        -1, 1
    ) @ spots_coverage.reshape(1, -1)

    _, known_base_nb_mean, _, _ = merge_pseudobulk_by_index_mix(
        single_X,
        known_single_base_nb_mean,
        np.zeros_like(known_single_base_nb_mean),
        initial_clone_index_baf,
        None,
    )

    return initial_clone_index_baf, known_base_nb_mean


def load_clone_ranges(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Clone ranges file not found: {filepath}")

    logger.info(f"Loading clone ranges from: {filepath.name}")

    # Read the TSV file
    df = pd.read_csv(filepath, sep="\t")

    # Standardize essential coordinate columns if they exist
    if "chr" in df.columns:
        # Cast to string to prevent int/str mismatches (e.g., 1 vs "1" vs "X")
        df["chr"] = df["chr"].astype(str)

    if "start" in df.columns:
        df["start"] = df["start"].astype(int)

    if "end" in df.columns:
        df["end"] = df["end"].astype(int)

    # Validate expected format
    expected_cols = {"chr", "start", "end"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        logger.warning(f"Clone ranges file is missing expected columns: {missing_cols}")

    logger.info(
        f"Loaded {len(df)} clone segments across {df['chr'].nunique()} chromosomes."
    )

    return df


def assign_clone_ranges(
    df_gene_snp, clone_ranges, key="known_id", collapse=True, max_length=1_000_000
):
    df_snps = df_gene_snp.copy()
    ranges_df = clone_ranges.copy()

    if collapse:
        state_cols = [c for c in ranges_df.columns if c not in ["chr", "start", "end"]]
        state_changed = (ranges_df[state_cols] != ranges_df[state_cols].shift(1)).any(
            axis=1
        )
        chr_changed = ranges_df["chr"] != ranges_df["chr"].shift(1)

        group_id = (state_changed | chr_changed).cumsum()

        agg_dict = {"chr": "first", "start": "first", "end": "last"}
        for col in state_cols:
            agg_dict[col] = "first"

        ranges_df = ranges_df.groupby(group_id).agg(agg_dict).reset_index(drop=True)

    if max_length is not None:
        new_rows = []
        for _, row in ranges_df.iterrows():
            start, end = row["start"], row["end"]
            if end - start > max_length:
                starts = np.arange(start, end, max_length)
                ends = np.clip(starts + max_length, a_min=None, a_max=end)

                for s, e in zip(starts, ends):
                    new_row = row.copy()
                    new_row["start"] = int(s)
                    new_row["end"] = int(e)
                    new_rows.append(new_row)
            else:
                new_rows.append(row)

        ranges_df = pd.DataFrame(new_rows).reset_index(drop=True)

    ranges_df[key] = np.arange(len(ranges_df))

    df_snps[key] = pd.NA

    for chrom, chrom_ranges in ranges_df.groupby("chr"):
        mask = df_snps["CHR"].astype(str) == str(chrom)
        if not mask.any():
            continue

        item_starts = df_snps.loc[mask, "START"].to_numpy()
        item_ends = df_snps.loc[mask, "END"].to_numpy()

        bin_starts = chrom_ranges["start"].to_numpy()
        bin_ends = chrom_ranges["end"].to_numpy()
        bin_ids = chrom_ranges[key].to_numpy()

        overlap_starts = np.maximum(item_starts[:, None], bin_starts[None, :])
        overlap_ends = np.minimum(item_ends[:, None], bin_ends[None, :])

        overlaps = np.maximum(0, overlap_ends - overlap_starts)

        best_bin_indices = np.argmax(overlaps, axis=1)

        max_overlaps = overlaps[np.arange(len(overlaps)), best_bin_indices]
        valid_mask = max_overlaps > 0

        valid_snps_idx = df_snps.index[mask][valid_mask]
        df_snps.loc[valid_snps_idx, key] = bin_ids[best_bin_indices[valid_mask]]

    df_snps[key] = df_snps[key].astype("Int64")

    return df_snps
