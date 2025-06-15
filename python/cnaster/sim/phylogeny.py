import copy
import random
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dataclasses import dataclass
from typing import Any, Optional

from cnaster_rs import ellipse, set_cnaster_rs_seed
from cnaster.config import JSONConfig

np.random.seed(42)
set_cnaster_rs_seed(42)

# TODO HACK
config = JSONConfig.from_file("/Users/mw9568/repos/cnaster/sim_config.json")
centers = (0.5 * np.array([[1, 1], [1, -1], [-1, 1]])).tolist()


@dataclass
class Node:
    identifier: int = -1
    time: int = -1
    cna_idx: int = -1
    ellipse_idx: int = -1
    parent: Optional["Node"] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    def __str__(self):
        return (
            f"Node(identifier={self.identifier}, time={self.time}, "
            f"cna_idx={self.cna_idx}, ellipse_idx={self.ellipse_idx})"
        )


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
        new_cna_idx = np.random.randint(0, len(current_cnas))
        new_cna = copy.deepcopy(current_cnas[new_cna_idx])
    else:
        state = random.choice(copy_num_states)
        pos = config.segment_size_kbp * np.random.randint(0, num_segments)

        new_cna = [state, pos, pos + config.cna_length_kbp]
        new_cna_idx = None

    return new_cna, new_cna_idx


def simulate_parent():
    center = np.random.uniform(
        low=-0.5, high=0.5, size=(2, 1)
    ).astype(float)

    inv_diag = np.array(
        np.random.randint(
            low=3, high=6, size=(2, 1)
        ),
        dtype=float,
    ).reshape(2, 1)

    theta = np.pi * np.random.randint(1, high=4) / 4.0

    return ellipse.CnaEllipse.from_inv_diagonal(center, inv_diag).rotate(theta)


def simulate_phylogeny():
    normal = Node(0, 0)
    tree = BinaryTree(normal)

    time = 1
    ellipses, parents, cnas = [], [], []

    node_count = 1

    while time < 5:
        leaf = tree.sample_leaf()
        cna_idx = leaf.cna_idx
        ellipse_idx = leaf.ellipse_idx

        # NB the normal leaf generates a metastasis
        if ellipse_idx == -1:
            print(f"\nTime {time}:  Solving for parent ellipse")

            if time == 1:
                center = np.array([-0.25, -0.25], dtype=float).reshape(2, 1)
                inv_diag = np.array([2.5, 2.5], dtype=float).reshape(2, 1)

                el = ellipse.CnaEllipse.from_inv_diagonal(center, inv_diag)
                el = el.rotate(np.pi / 4.0)
            else:
                while True:
                    el = simulate_parent()

                    valid = True
                    
                    for parent in parents:
                        if parent.overlaps(el):
                            valid = False
                            break
                    
                    if valid:
                        break

            parents.append(el)
        else:
            print(f"\nTime {time}:  Solving for daughter ellipse")

            el = ellipses[ellipse_idx].get_daughter(0.75)

        print(f"Time {time}:  Solving for lineage")

        lineage_cna_idxs = [leaf.cna_idx]
        parent = leaf.parent

        while parent is not None:
            lineage_cna_idxs.append(parent.cna_idx)
            parent = parent.parent

        print(f"Time {time}:  Solving for CNA")

        while True:
            cna, new_cna_idx = simulate_cna(cnas, config.phylogeny.parsimony_rate)

            if new_cna_idx is not None:
                if new_cna_idx not in lineage_cna_idxs:
                    break
            else:
                break

        ellipses.append(el)

        print(f"Time {time}:  Solved for CNA:  {cna}")

        if new_cna_idx is None:
            cnas.append(cna)
            new_cna_idx = len(cnas) - 1

        print(f"Time {time}:  Solving for children")

        leaf.left = Node(leaf.identifier, time, leaf.cna_idx, leaf.ellipse_idx, leaf)
        leaf.right = Node(node_count, time + 0.5, new_cna_idx, len(ellipses) - 1, leaf)

        print(f"\n\n{leaf.left}\n{leaf.right}")

        node_count += 1
        time += 1

    return tree, ellipses, cnas


def plot_ellipse(center, L, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    t = np.linspace(0, 2 * np.pi, 100)
    circle = np.stack([np.cos(t), np.sin(t)])

    ellipse = L @ circle
    ellipse = ellipse + np.array(center).reshape(2, 1)

    ax.plot(ellipse[0], ellipse[1], **kwargs)

    return ax


def plot_phylogeny(tree, ellipses, cnas):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    root = tree.root
    leaves = tree.leaves()

    max_time = max(leaf.time for leaf in leaves)

    def plot_node(node, xshift=0.0):
        if node is None:
            return

        # NB plot ellipse
        if node.ellipse_idx >= 0:
            el = ellipses[node.ellipse_idx]
            center = el.center
            L = el.l

            plot_ellipse(
                center, L, ax=axes[0], color=colors[1 + node.cna_idx], alpha=0.25
            )

        # NB plot tree
        axes[1].scatter(xshift, -node.time, color=colors[1 + node.cna_idx])

        right_dx = 0.025 * np.random.randint(0, 5)

        if node.left is not None:
            axes[1].plot(
                [xshift, xshift],
                [-node.time, -node.left.time],
                color=colors[1 + node.cna_idx],
                alpha=0.5,
            )

        if node.right is not None:
            axes[1].plot(
                [xshift, xshift + 1.0 + right_dx],
                [-node.left.time, -node.right.time],
                color=colors[1 + node.right.cna_idx],
                alpha=0.5,
            )

            axes[1].text(
                np.mean([xshift, xshift + 1.0 + right_dx]),
                0.99 * np.mean([-node.left.time, -node.right.time]),
                f"{cnas[node.right.cna_idx][0]}",
                fontsize=8,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc=None, alpha=0.7, lw=0),
            )

        if node.left is None and node.right is None:
            axes[1].scatter(xshift, -max_time, color=colors[1 + node.cna_idx])
            axes[1].plot(
                [xshift, xshift],
                [-node.time, -max_time],
                color=colors[1 + node.cna_idx],
                alpha=0.5,
            )

        plot_node(node.left, xshift)
        plot_node(node.right, xshift + 1.0 + right_dx)

    plot_node(root)

    axes[1].text(
        0.05,
        0.0,
        f"Normal",
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
    )

    axes[0].set_xlim(-0.5, 0.5)
    axes[0].set_ylim(-0.5, 0.5)

    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    axes[1].set_yticks([])
    axes[1].set_yticklabels([])

    axes[1].set_xlabel("Number of CNAs")
    axes[1].set_ylabel("Look-back time")

    plt.show()


if __name__ == "__main__":
    while True:
        tree, ellipses, cnas = simulate_phylogeny()
        plot_phylogeny(tree, ellipses, cnas)

    print("\n\nDone.\n\n")
