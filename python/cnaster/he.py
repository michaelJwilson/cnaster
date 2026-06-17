import matplotlib.pyplot as plt
import numpy as np
import json
import polars as pl

from pathlib import Path
from scipy.spatial import cKDTree
from cnaster.logger import get_logger
from cnaster.config import start_time

logger = get_logger(__name__)

def join_tables_xy(
    first: pl.DataFrame, second: pl.DataFrame, columns_to_merge: list[str]
) -> pl.DataFrame:
    """
    Joins second to first with {columns_to_merge}, such that the match in
    second is closest in (x,y) to the instance of first.
    """
    first_xy = np.column_stack([first["x"].to_numpy(), first["y"].to_numpy()])
    second_xy = np.column_stack([second["x"].to_numpy(), second["y"].to_numpy()])

    tree = cKDTree(second_xy)

    # NB for each row in first, find closest in second.
    distances, indices = tree.query(first_xy)

    new_columns = [
        pl.Series(col, second[col].to_numpy()[indices]) for col in columns_to_merge
    ]

    new_columns = new_columns + [pl.Series("dist", distances)]

    return first.with_columns(new_columns)

def get_he_image(spaceranger_dir, res="hires", pos=None, num_labels=4):
    assert res in ("lowres", "hires")
    # scalefactor = target_size / max (original image height, original image width),
    #
    # e.g. {
    #    "spot_diameter_fullres": 58.45684684229273,
    #    "bin_size_um": 16.0,
    #    "microns_per_pixel": 0.2737061758251425,
    #    "tissue_lowres_scalef": 0.0071874363,
    #    "fiducial_diameter_fullres": 1205.6724661222877,
    #    "tissue_hires_scalef": 0.071874365,
    #    "regist_target_img_scalef": 0.071874365
    # }
    sf_path = Path(f"{spaceranger_dir}/spatial/scalefactors_json.json")
    img_path = Path(f"{spaceranger_dir}/spatial/tissue_{res}_image.png")

    if not sf_path.exists() or not img_path.exists():
        logger.warning(
            f"Could not find H&E image or scalefactors at {spaceranger_dir}/spatial/"
        )
        return pos

    with open(sf_path, "r") as ff:
        scalefactors = json.load(ff)

    scalefactor = scalefactors[f"tissue_{res}_scalef"]

    # NB realizes a (H, W, C) numpy array, i.e. (382, 600, 3) for low and (3818, 6000, 3) for high (HD @ 6.5mm).
    tissue_image = plt.imread(img_path)

    # NB 11,222,500 rows in the 2 um version of the file for Visium HD
    # full_width = np.ceil(6000 / 0.071874365) = 83_479.0
    # full_height = np.ceil(3818 / 0.071874365) = 53_121.0

    # NB barcode positions correspond to 5_641.821 < y < 30_099.837, 1_190.55 < x < 5_641.82 for 175_561 spots.
    #    i.e. the aligned portion of the original image.
    H, W, C = tissue_image.shape
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    tissue_frame = pl.DataFrame(
        {
            "red": tissue_image[:, :, 0].flatten(),
            "green": tissue_image[:, :, 1].flatten(),
            "blue": tissue_image[:, :, 2].flatten(),
            "array_row": rows.flatten(),
            "array_col": cols.flatten(),
        }
    )

    # NB 0.0 < x < 83_339.869; 0 < y < 53_009.165
    tissue_frame = tissue_frame.with_columns(
        (pl.col("array_col") / scalefactor).alias("x"),
        (pl.col("array_row") / scalefactor).alias("y"),
    )

    if pos is not None:
        # NB limited to in_tissue=True
        pos = pl.from_pandas(pos)

        columns = ("red", "green", "blue")
        tissue_frame = join_tables_xy(pos, tissue_frame, columns)

    def crop_values(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    rgb = tissue_frame.select(["red", "green", "blue"]).to_numpy()

    # TODO standard lib?
    gray = 0.2125 * rgb[:, 0] + 0.7154 * rgb[:, 1] + 0.0721 * rgb[:, 2]
    cropped_gray = crop_values(gray)

    percentiles = np.linspace(0.0, 100.0, 1 + num_labels)
    bins = np.percentile(np.sort(cropped_gray.flatten()), percentiles)

    labels = np.digitize(cropped_gray, bins=bins)

    # TODO rename he_label.
    # NB 0.0 < x < 83_339.869; 0 < y < 53_009.165
    tissue_frame = tissue_frame.with_columns(
        pl.Series("gray", gray),
        pl.Series("cropped_gray", cropped_gray),
        pl.Series("label", labels),
    )

    logger.info(f"Merged with h&e with result:\n{tissue_frame}")

    return tissue_frame.to_pandas()
