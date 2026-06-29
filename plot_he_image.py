from cnaster.config import YAMLConfig, set_global_config, start_time
from cnaster.utils import configure_output_dir, write_fig
from cnaster.logger import get_logger
from cnaster.he import get_he_image
from cnaster.plotting import plot_he


logger = get_logger(__name__, start_time=start_time)

# TODO HACK
config_path = "/u/mw9568/research/repos/cnaster/config.yaml"

config = YAMLConfig.from_file(config_path)
config.issue_warnings()

logger.info(f"Read configuration:\n{config}")

set_global_config(config)

output_dir, plots_dir = configure_output_dir(config)

# TODO HACK
sample_id = "N3KXRGQ"

# TODO constructor:  defined for all spots, in_tissue, etc?
spaceranger_dir = f"/u/mw9568/data/{sample_id}/output_16um"

he_frame = get_he_image(
    spaceranger_dir=spaceranger_dir, res="lowres", pos=None, num_labels=4
)

he_fig = plot_he(
    he_frame,
    channels=("image",),
)

opath = f"{plots_dir}/he_image.svg"

write_fig(
    opath,
    he_fig,
    transparent=True,
    bbox_inches="tight",
    format="svg",
    dpi=300,
)
