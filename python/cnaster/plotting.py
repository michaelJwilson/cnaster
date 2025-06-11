import matplotlib.pyplot as plt


def plot_cnaster_graph(cnaster_graph):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    labels, positions = cnaster_graph.labels, cnaster_graph.positions

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=labels)

    for from_idx, to_idx in edges:
        x = [sites[from_idx, 0], sites[to_idx, 0]]
        y = [sites[from_idx, 1], sites[to_idx, 1]]
        z = [sites[from_idx, 2], sites[to_idx, 2]]

        ax.plot(x, y, z, color="black", linewidth=0.5, alpha=0.5)

        normed_edge_weights = alignment_weights / alignment_weights.max()

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
