import argparse
import logging
import time
from pathlib import Path
from cnaster.config import JSONConfig
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def run_sim(config_path, debug=True):
    start = time.time()

    config = JSONConfig.from_file(config_path)

    output_dir = Path(config.output_dir)

    run_id = 00000 if debug else datetime.now().strftime('%Y%m%d_%H%M%S')
    
    run_dir = output_dir / f"run{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for sample_name, sample_info in config.samples.items():
        logger.info(f"Solving for sample {sample_name}")
        
        sample_dir = run_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created sample directory {sample_dir}")
        
    """
    for sample in config.samples:        

        Path(f"{output_dir}/cna_sim_{sim_id}/plots").mkdir(exist_ok=True, parents=True)

        cna_sim = CNA_sim(sim_id=sim_id, seed=seed + sim_id)
        cna_sim.save(output_dir)

        cna_sim.plot_realization_true_flat(
            f"{output_dir}/cna_sim_{sim_id}/plots/truth_rdr_baf_flat_{sim_id}.pdf"
        )
        cna_sim.plot_realization_true_genome(
            f"{output_dir}/cna_sim_{sim_id}/plots/truth_rdr_baf_genome_{sim_id}.pdf"
        )
    """

    logger.info(f"\n\nDone ({time.time() - start:.3f} seconds).\n\n")


def main():
    # NB run_cnaster_sim --config_path /Users/mw9568/repos/cnaster/sim_config.json
    parser = argparse.ArgumentParser(description="Create CNA simulation.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        required=True,
        help="Path to sim. configuration file",
    )

    args = parser.parse_args()

    run_sim(args.config_path)


if __name__ == "__main__":
    main()
