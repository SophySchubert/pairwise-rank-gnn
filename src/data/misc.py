import networkx as nx
from scipy.stats import kendalltau
import numpy as np
from spektral.data.loaders import tf_loader_available
from spektral.data.utils import (
    prepend_none,
    sp_matrices_to_sp_tensors,
    to_disjoint,
    collate_labels_disjoint,
    batch_generator
)


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



class FrankensteinLoader:

    def __init__(self, dataset, pairs_and_target, config, node_level=False, batch_size=1, epochs=None, shuffle=True):
        self.dataset = dataset
        self.pairs_and_target = pairs_and_target
        self.config = config
        self.node_level = node_level
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.seed = config['seed']
        self._generator = self.generator()

    def __iter__(self):
        return self

    def __next__(self):
        nxt = self._generator.__next__()
        return self.collate(nxt)

    def generator(self):
        return batch_generator(
            self.pairs_and_target,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=self.shuffle,
        )

    def collate(self, batch):
        print(f"batch:{batch}")
        idx_a, idx_b, target = zip(*[(x[0], x[1], x[2]) for x in batch])
        batch = self.get_batch_data(idx_a, idx_b)
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_disjoint(y, node_level=self.node_level)

        output = to_disjoint(**packed)
        output = sp_matrices_to_sp_tensors(output)
        print(f"output:{output + (idx_a, idx_b)}")

        return output + (idx_a, idx_b), target

    def get_batch_data(self, idx_a, idx_b, mode="default"):
        required_indices = np.array(list(set(idx_a + idx_b)))
        required_data = self.dataset[required_indices]
        if mode == "default":
            return required_data
        elif mode == "type-vstack":
            pass # see notebook
        elif mode == "type-hstack":
            pass # see notebook
        elif mode == "type-merge":
            pass # see notebook

        return None


    def load(self):
        # TODO: maybe replace with function from DisjointLoader
        return self

    def tf_signature(self):
        # TODO: maybe replace with function from DisjointLoader
        signature = self.dataset.signature
        return to_tf_signature(signature)

    def pack(self, batch):
        output = [list(elem) for elem in zip(*[g.numpy() for g in batch])]
        keys = [k + "_list" for k in self.dataset.signature.keys()]
        return dict(zip(keys, output))

    @property
    def steps_per_epoch(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))