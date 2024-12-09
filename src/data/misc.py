import networkx as nx
import numpy as np


def spektral_graph_to_nx_graph(spektral_graph):
    # Create a NetworkX graph from the adjacency matrix
    g = nx.from_scipy_sparse_matrix(spektral_graph.a)

    # Add node features to the NetworkX graph
    for i, features in enumerate(spektral_graph.x):
        g.nodes[i]['features'] = features

    return g

