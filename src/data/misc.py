import networkx as nx
from scipy.stats import kendalltau

def spektral_graph_to_nx_graph(spektral_graph):
    # Create a NetworkX graph from the adjacency matrix
    g = nx.from_scipy_sparse_matrix(spektral_graph.a)

    # Add node features to the NetworkX graph
    for i, features in enumerate(spektral_graph.x):
        g.nodes[i]['features'] = features

    return g

def compare_rankings(ranking_a, ranking_b):
    # Extract the rankings from the sorted tuples
    ranking_a = [index for graph, index in ranking_a]
    ranking_b = [index for graph, index in ranking_b]

    # Compute the Kendall Tau distance
    correlation_coefficient, p_value = kendalltau(ranking_a, ranking_b)
    return correlation_coefficient, p_value



