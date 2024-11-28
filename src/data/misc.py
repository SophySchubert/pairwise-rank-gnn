import networkx as nx
import pickle


def spektral_to_networkx(spektral_graph):
    # Create a NetworkX graph from the adjacency matrix
    nx_graph = nx.from_scipy_sparse_matrix(spektral_graph.a)

    # Add node features to the NetworkX graph
    for i, features in enumerate(spektral_graph.x):
        nx_graph.nodes[i]['features'] = features

    return nx_graph

def saveGraph(graph):
    with open("temp.p", "wb") as f:
        pickle.dump(graph, f)

def loadGraph(name:'temp.p'):
    with open("temp.p", "rb") as f:
        return pickle.load(f)

if "name" == "__main__":
    pass