import numpy as np

from cnaster_rs import ellipse
from cnaster.config import SimConfig


def get_sim_cna(copy_states, current_cnas, parsimony_rate):
    
    
    
def create_phylogeny():
    # TODO HACK
    sim_config = SimConfig.from_file("/Users/mw9568/repos/cnaster/sim_config.json")
    copy_num_states = np.array(sim_config.copy_num_states)
    
    # NB define parent ellipse.
    center = np.array([0.0, 0.0]).reshape((2, 1))
    inv_diag = np.array([1.0, 2.0]).reshape((2, 1))

    el = ellipse.CnaEllipse.from_diagonal(center, inv_diag)
    el = el.rotate(np.pi / 4.0)

    
    
    cnas = []
    hierarchy = []

    for ii in range(3):
        el = el.get_daughter(0.75)
        hierarchy.append(el)


if __name__ == "__main__":
    create_phylogeny()
