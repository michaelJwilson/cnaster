import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dataclasses import dataclass
from typing import Any, Optional

from cnaster_rs import ellipse
from cnaster.config import JSONConfig

# TODO HACK
config = JSONConfig.from_file("/Users/mw9568/repos/cnaster/sim_config.json")
centers = [[0, 0], [5, 5], [5, -5], [-5, -5], [-5, 5]]

@dataclass
class Node:
    time: int = -1
    cna_idx: int = -1
    ellipse_idx: int = -1
    parent: Optional["Node"] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None


class BinaryTree:
    def __init__(self, root: Optional[Node] = None):
        self.root = root

    def leaves(self) -> list:
        def _leaves(node):
            if node is None:
                return []
            if node.left is None and node.right is None:
                return [node]

            return _leaves(node.left) + _leaves(node.right)

        return _leaves(self.root)

    def sample_leaf(self) -> Optional[Node]:
        leaves = self.leaves()

        if not leaves:
            return None

        return random.choice(leaves)


def simulate_cna(current_cnas, parsimony_rate):
    copy_num_states = np.array(config.copy_num_states)
    num_segments = config.mappable_genome_kbp // config.segment_size_kbp

    if current_cnas and (np.random.uniform() < parsimony_rate):
        new_cna = np.random.choice(current_cnas)
        return new_cna, current_cnas.index(new_cna)
    else:
        state = random.choice(copy_num_states)
        pos = config.segment_size_kbp * np.random.randint(0, num_segments)
    
        return [state, pos, pos + config.cna_length_kbp], None


def simulate_parent():
    center = np.array(centers.pop(), dtype=float).reshape(2, 1)
    inv_diag = np.array([1.0, np.random.randint(1, high=4)], dtype=float).reshape(2, 1)

    theta = np.pi * np.random.randint(1, high=4) / 4.0

    return ellipse.CnaEllipse.from_diagonal(center, inv_diag).rotate(theta)


def simulate_phylogeny():
    normal = Node(0)
    tree = BinaryTree(normal)

    time = 1
    ellipses, cnas = [], []

    while time < 3:
        leaf = tree.sample_leaf()
        cna_idx = leaf.cna_idx
        ellipse_idx = leaf.ellipse_idx

        # NB the normal leaf generates a metastasis
        if ellipse_idx == -1:
            if time == 0:
                center = np.array([0.0, 0.0], dtype=float).reshape(2, 1)
                inv_diag = np.array([1.0, 2.0], dtype=float).reshape(2, 1)

                el = ellipse.CnaEllipse.from_diagonal(center, inv_diag)
                el = el.rotate(np.pi / 4.0)
            else:
                el = simulate_parent()
        else:
            el = ellipses[ellipse_idx].get_daughter(0.75)

        lineage_cna_idxs = [leaf.cna_idx]
        parent = leaf.parent

        while parent is not None:
            lineage_cna_idxs.append(parent.cna_idx)
            leaf = parent

        while True:
            cna, new_cna_idx = simulate_cna(cnas, config.phylogeny.parsimony_rate)

            if new_cna_idx is not None:
                if new_cna_idx not in lineage_cna_idxs:
                    break
            else:
                break

        ellipses.append(el)

        if new_cna_idx is None:
            cnas.append(cna)
            new_cna_idx = cna_idx + 1

        leaf.left = copy.deepcopy(leaf)
        leaf.right = Node(time, new_cna_idx, ellipse_idx + 1, leaf)

        time += 1

    return tree, ellipses, cnas

def plot_phylogeny(tree, ellipses, cnas):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    leaves = tree.leaves()

    for leaf in leaves:
        if leaf.ellipse_idx != -1:
            el = ellipses[leaf.ellipse_idx]

            axes[0].scatter(
                el.center[0][0],
                el.center[0][0],
            )

    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    tree, ellipses, cnas = simulate_phylogeny()

    plot_phylogeny(tree, ellipses, cnas)