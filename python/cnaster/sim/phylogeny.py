import copy
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional

from cnaster_rs import ellipse
from cnaster.config import JSONConfig

# TODO HACK
config = JSONConfig.from_file("/Users/mw9568/repos/cnaster/sim_config.json")
cna_id = 0

centers = [[0,0], [5,5], [5,-5], [-5,-5], [-5,5]]

@dataclass
class Node:
    time: int = -1
    ellipse: Optional['ellipse.CnaEllipse'] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None

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
        
        return np.random.choice(leaves)
    
def simulate_cna(current_cnas, parsimony_rate):
    copy_num_states = np.array(config.copy_num_states)
    num_segments = config.mappable_genome_kbp // config.segment_size_kbp

    if current_cnas and (np.random.uniform() < parsimony_rate):
        # NB  randomly select an existing copy number state.
        return np.random.choice(current_cnas, size=1, replace=True)
    else:
        state = np.random.choice(copy_num_states, size=1, replace=True)
        pos = config.segment_size_kbp * np.random.randint(0, num_segments, size=1)

        new_cna = [cna_id, state, pos, config.cna_length_kbp]
        cna_id += 1
        
        return new_cna

def simulate_parent():
    center = np.array(centers.pop(), dtype=float).reshape(2, 1)
    inv_diag = np.array([1.0, np.random.randint(1, high=4)], dtype=float).reshape(2, 1)

    theta = np.pi * np.random.randint(1, high=4) / 4.0

    return ellipse.CnaEllipse.from_diagonal(center, inv_diag).rotate(theta)
                         
    
def simulate_phylogeny():
    # NB initialize normal root at time zero.
    normal = Node(0)                         
    tree = BinaryTree(normal)

    time = 1

    while time < 2:
        leaf = tree.sample_leaf()

        # cna = simulate_cna(cnas, config.phylogeny.parsimony_rate)
                         
        # NB the normal leaf generates a new tumor of sub-clones.
        if leaf.ellipse == None:
            if time == 0:
                center = np.array([0.0, 0.0], dtype=float).reshape(2, 1)
                inv_diag = np.array([1.0, 2.0], dtype=float).reshape(2, 1)

                el = ellipse.CnaEllipse.from_diagonal(center, inv_diag)
                el = el.rotate(np.pi / 4.0)
            else:
                el = simulate_parent()
        else:
            el = leaf.ellipse.get_daughter(0.75)

        print(el)

        leaf.left = copy.deepcopy(leaf)                 
        # leaf.right = Node(time, el)
        
        time += 1


if __name__ == "__main__":
    simulate_phylogeny()