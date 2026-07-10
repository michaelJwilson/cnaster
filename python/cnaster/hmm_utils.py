from cnaster.config import get_global_config, start_time
from cnaster.count_encoder import CountEncoder
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

# NB define global alias for legacy
construct_unique_matrix = CountEncoder.construct_unique_encoding


def get_solver():
    known_solvers = ("BFGS", "L-BFGS-B", "Nelder-Mead")

    name = get_global_config().hmm.solver

    assert (
        name in known_solvers
    ), f"Unknown solver: {name}. Supported solvers: {known_solvers}"

    return name


def get_em_solver_params():
    """
    Get the parameters for the emission solver.
    """
    config = get_global_config()
    solver = config.hmm.solver

    match solver:
        case "BFGS":
            kwargs = ("xrtol", "disp")
        case "L-BFGS-B":
            kwargs = ("maxiter", "ftol", "disp")
        case "Nelder-Mead":
            kwargs = ("maxiter", "xtol", "ftol", "disp")
        case _:
            raise ValueError(f"cnaster does not support solver: {solver}")

    return {k.replace("em_", ""): float(getattr(config.hmm, f"em_{k}")) for k in kwargs}
