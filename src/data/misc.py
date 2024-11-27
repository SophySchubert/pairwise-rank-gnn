import networkx as nx
from node2vec import Node2Vec
import pickle


def spektral_to_networkx(spektral_graph):
    # Create a NetworkX graph from the adjacency matrix
    nx_graph = nx.from_scipy_sparse_matrix(spektral_graph.a)

    # Add node features to the NetworkX graph
    for i, features in enumerate(spektral_graph.x):
        nx_graph.nodes[i]['features'] = features

    return nx_graph

def graph2embedding():
    # TODO: expand to work with real dataset
    EMBEDDING_FILENAME = "temp.txt"
    EMBEDDING_MODEL_FILENAME = "temp.emb"

    graph = nx.fast_gnp_random_graph(n=100, p=0.5)

    # print(graph.nodes())
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs
    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Look for most similar nodes
    model.wv.most_similar('2')  # Output node names are always strings

    # Save embeddings for later use
    model.wv.save_word2vec_format(EMBEDDING_FILENAME)

    # Save model for later use
    # model.save(EMBEDDING_MODEL_FILENAME)

def saveGraph(graph):
    with open("temp.p", "wb") as f:
        pickle.dump(graph, f)

def loadGraph(name:'temp.p'):
    with open("temp.p", "rb") as f:
        return pickle.load(f)