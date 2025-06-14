import numpy as np

from cnaster_rs import ellipse
from cnaster.config import JSONConfig

# TODO HACK
config = JSONConfig.from_file("/Users/mw9568/repos/cnaster/sim_config.json")


def simulate_cna(current_cnas, parsimony_rate):
    copy_num_states = np.array(config.copy_num_states)
    num_segments = config.mappable_genome_kbp // config.segment_size_kbp

    if current_cnas and (np.random.uniform() < parsimony_rate):
        # NB  randomly select an existing copy number state.
        return current_cnas + np.random.choice(current_cnas, size=1, replace=True)
    else:
        state = np.random.choice(copy_num_states, size=1, replace=True)
        pos = config.segment_size_kbp * np.random.randint(0, num_segments, size=1)

        return current_cnas + [[state, pos, config.cna_length_kbp]]


def simulate_phylogeny():
    # NB define parent ellipse.
    center = np.array([0.0, 0.0]).reshape((2, 1))
    inv_diag = np.array([1.0, 2.0]).reshape((2, 1))

    el = ellipse.CnaEllipse.from_diagonal(center, inv_diag)
    el = el.rotate(np.pi / 4.0)

    hierarchy = [el]
    cnas = simulate_cna([], config.phylogeny.parsimony_rate)

    for ii in range(3):
        el = el.get_daughter(0.75)
        hierarchy.append(el)

        cnas = simulate_cna(cnas, config.phylogeny.parsimony_rate)

    for cna in cnas:
        print(f"Copy number state: {cna[0]}, Position: {cna[1]} kbp, Length: {cna[2]} kbp")

if __name__ == "__main__":
    simulate_phylogeny()