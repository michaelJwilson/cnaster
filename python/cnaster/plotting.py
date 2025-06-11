import matplotlib.pyplot as plt


def plot_cnaster_graph(cnaster_graph, alpha=0.1):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    max_weight = cnaster_graph.max_edge_weight
    labels, sites = cnaster_graph.labels, cnaster_graph.positions

    ax.scatter(sites[:, 0], sites[:, 1], sites[:, 2], c=labels)

    for from_idx, edges in cnaster_graph.adjacency_list.items():        
        for edge in edges:
            to_idx, weight = edge

            x = [sites[from_idx, 0], sites[to_idx, 0]]
            y = [sites[from_idx, 1], sites[to_idx, 1]]
            z = [sites[from_idx, 2], sites[to_idx, 2]]

            weight = weight if weight == 1. else weight / max_weight
            
            ax.plot(x, y, z, color="gray", linewidth=0.5, alpha=alpha * weight)
    
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
