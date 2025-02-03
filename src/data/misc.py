import networkx as nx
from scipy.stats import kendalltau
import numpy as np

from itertools import combinations

from spektral.data.loaders import tf_loader_available
from spektral.data.utils import (
    prepend_none,
    sp_matrices_to_sp_tensors,
    to_disjoint,
    collate_labels_disjoint,
    batch_generator
)

from spektral.data import Graph


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

def sample_preference_pairs2(graphs):
    c = [(a, b, check_util2(graphs, a,b)) for a, b in combinations(range(len(graphs)), 2)]
    return np.array(c)

def check_util2(data, index_a, index_b):
    a = data[index_a]
    b = data[index_b]
    util_a = a.y
    util_b = b.y
    if util_a >= util_b:
        return 1
    else:
        return 0


class CustomDisjointedLoader:

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
        if mode == "default":
            required_indices = np.array(list(set(idx_a + idx_b)))
            required_data = self.dataset[required_indices]
            return required_data
        elif mode == "type-vstack":
            return merge_vstack(idx_a, idx_b, self.dataset, self.config)
        elif mode == "type-hstack":
           return merge_hstack(idx_a, idx_b, self.dataset, self.config)
        elif mode == "type-merge-mult":
           return merge_mult(idx_a, idx_b, self.dataset, self.config)
        elif mode == "type-merge-add":
            return merge_add(idx_a, idx_b, self.dataset, self.config)
        elif mode == "type-merge-mean":
            return merge_mean(idx_a, idx_b, self.dataset, self.config)
        else:
            raise ValueError(f"Mode {mode} unknown")

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

def merge_vstack(idx_a, idx_b, dataset, config):
    # mode=="type-vstack"
    assert(len(idx_a)==len(idx_b))
    graphs = []
    for i in range(len(idx_a)):
        g_1, g_2 = dataset[idx_a[i]], dataset[idx_b[i]]
        assert(g_1.x.shape[1]==config['x_shape1'] and g_2.x.shape[1]==config['x_shape1'])
        assert(g_2.e.shape[1]==3 and g_2.e.shape[1]==3)

        if not (g_1.a.shape>g_2.a.shape or g_1.a.shape<g_2.a.shape):
            a = np.vstack((g_1.a, g_2.a))
        elif (g_1.a.shape > g_2.a.shape):
            _a = np.pad(g_2.a, ((0, g_1.a.shape[0]-g_2.a.shape[0]), (0, g_1.a.shape[1]-g_2.a.shape[1])), mode='constant', constant_values=0)
            a = np.vstack((g_1.a, _a))
        elif (g_1.a.shape < g_2.a.shape):
            _a = np.pad(g_1.a, ((0, g_2.a.shape[0]-g_1.a.shape[0]), (0, g_2.a.shape[1]-g_1.a.shape[1])), mode='constant', constant_values=0)
            a = np.vstack((_a, g_2.a))

        g_n = Graph()
        g_n.x = np.vstack((g_1.x, g_2.x))
        g_n.e = np.vstack((g_1.e, g_2.e))
        g_n.a = a
        graphs.append(g_n)

    return graphs

def merge_hstack(idx_a, idx_b, dataset, config):
    # mode=="type-hstack"
    assert(len(idx_a)==len(idx_b))
    graphs = []
    for i in range(len(idx_a)):
        g_1, g_2 = dataset[idx_a[i]], dataset[idx_b[i]]
        assert(g_1.x.shape[1]==config['x_shape1'] and g_2.x.shape[1]==config['x_shape1'])
        assert(g_2.e.shape[1]==3 and g_2.e.shape[1]==3)
        #a
        if not (g_1.a.shape > g_2.a.shape or g_1.a.shape < g_2.a.shape):
            a = np.hstack((g_1.a, g_2.a))
        elif (g_1.a.shape > g_2.a.shape):
            _a = np.pad(g_2.a, ((0, g_1.a.shape[0]-g_2.a.shape[0]), (0, g_1.a.shape[1]-g_2.a.shape[1])), mode='constant', constant_values=0)
            a = np.hstack((g_1.a, _a))
        elif (g_1.a.shape < g_2.a.shape):
            _a = np.pad(g_1.a, ((0, g_2.a.shape[0]-g_1.a.shape[0]), (0, g_2.a.shape[1]-g_1.a.shape[1])), mode='constant', constant_values=0)
            a = np.hstack((_a, g_2.a))
        #x
        if not (g_1.x.shape > g_2.x.shape or g_1.x.shape < g_2.x.shape):
            x = np.hstack((g_1.x, g_2.x))
        elif (g_1.x.shape > g_2.x.shape):
            _x = np.pad(g_2.x, ((0, g_1.x.shape[0]-g_2.x.shape[0]), (0, g_1.x.shape[1]-g_2.x.shape[1])), mode='constant', constant_values=0)
            x = np.hstack((g_1.x, _x))
        elif (g_1.x.shape < g_2.x.shape):
            _x = np.pad(g_1.x, ((0, g_2.x.shape[0]-g_1.x.shape[0]), (0, g_2.x.shape[1]-g_1.x.shape[1])), mode='constant', constant_values=0)
            x = np.hstack((_x, g_2.x))
            #e
        if not (g_1.e.shape > g_2.e.shape or g_1.e.shape < g_2.e.shape):
            e = np.hstack((g_1.e, g_2.e))
        elif (g_1.e.shape > g_2.e.shape):
            _e = np.pad(g_2.e, ((0, g_1.e.shape[0]-g_2.e.shape[0]), (0, g_1.e.shape[1]-g_2.e.shape[1])), mode='constant', constant_values=0)
            e = np.hstack((g_1.e, _e))
        elif (g_1.e.shape < g_2.e.shape):
            _e = np.pad(g_1.e, ((0, g_2.e.shape[0]-g_1.e.shape[0]), (0, g_2.e.shape[1]-g_1.e.shape[1])), mode='constant', constant_values=0)
            e = np.hstack((_e, g_2.e))

        g_n = Graph()
        g_n.x = x
        g_n.e = e
        g_n.a = a
        graphs.append(g_n)

    return graphs

def merge_mult(idx_a, idx_b, dataset, config):
    # mode=="type-merge-mult"
    assert(len(idx_a)==len(idx_b))
    graphs = []
    for i in range(len(idx_a)):
        g_1, g_2 = dataset[idx_a[i]], dataset[idx_b[i]]
        assert(g_1.x.shape[1]==config['x_shape1'] and g_2.x.shape[1]==config['x_shape1'])
        assert(g_2.e.shape[1]==3 and g_2.e.shape[1]==3)
        #a
        if not (g_1.a.shape > g_2.a.shape or g_1.a.shape < g_2.a.shape):
            a = g_1.a * g_2.a
        elif (g_1.a.shape > g_2.a.shape):
            _a = np.pad(g_2.a, ((0, g_1.a.shape[0]-g_2.a.shape[0]), (0, g_1.a.shape[1]-g_2.a.shape[1])), mode='constant', constant_values=0)
            a = g_1.a * _a
        elif (g_1.a.shape < g_2.a.shape):
            _a = np.pad(g_1.a, ((0, g_2.a.shape[0]-g_1.a.shape[0]), (0, g_2.a.shape[1]-g_1.a.shape[1])), mode='constant', constant_values=0)
            a = _a * g_2.a
        #x
        if not (g_1.x.shape > g_2.x.shape or g_1.x.shape < g_2.x.shape):
            x = g_1.x * g_2.x
        elif (g_1.x.shape > g_2.x.shape):
            _x = np.pad(g_2.x, ((0, g_1.x.shape[0]-g_2.x.shape[0]), (0, g_1.x.shape[1]-g_2.x.shape[1])), mode='constant', constant_values=0)
            x = g_1.x * _x
        elif (g_1.x.shape < g_2.x.shape):
            _x = np.pad(g_1.x, ((0, g_2.x.shape[0]-g_1.x.shape[0]), (0, g_2.x.shape[1]-g_1.x.shape[1])), mode='constant', constant_values=0)
            x = _x * g_2.x
        #e
        if not (g_1.e.shape > g_2.e.shape or g_1.e.shape < g_2.e.shape):
            e = g_1.e * g_2.e
        elif (g_1.e.shape > g_2.e.shape):
            _e = np.pad(g_2.e, ((0, g_1.e.shape[0]-g_2.e.shape[0]), (0, g_1.e.shape[1]-g_2.e.shape[1])), mode='constant', constant_values=0)
            e = g_1.e * _e
        elif (g_1.e.shape < g_2.e.shape):
            _e = np.pad(g_1.e, ((0, g_2.e.shape[0]-g_1.e.shape[0]), (0, g_2.e.shape[1]-g_1.e.shape[1])), mode='constant', constant_values=0)
            e = _e * g_2.e
        g_n = Graph()
        g_n.a = a
        g_n.x = x
        g_n.e = e
        graphs.append(g_n)

    return graphs

def merge_add(idx_a, idx_b, dataset, config):
    # mode=="type-merge-add"
    assert(len(idx_a)==len(idx_b))
    graphs = []
    for i in range(len(idx_a)):
        g_1, g_2 = dataset[idx_a[i]], dataset[idx_b[i]]
        assert(g_1.x.shape[1]==config['x_shape1'] and g_2.x.shape[1]==config['x_shape1'])
        assert(g_2.e.shape[1]==3 and g_2.e.shape[1]==3)
        #a
        if not (g_1.a.shape > g_2.a.shape or g_1.a.shape < g_2.a.shape):
            a = g_1.a + g_2.a
        elif (g_1.a.shape > g_2.a.shape):
            _a = np.pad(g_2.a, ((0, g_1.a.shape[0]-g_2.a.shape[0]), (0, g_1.a.shape[1]-g_2.a.shape[1])), mode='constant', constant_values=0)
            a = g_1.a + _a
        elif (g_1.a.shape < g_2.a.shape):
            _a = np.pad(g_1.a, ((0, g_2.a.shape[0]-g_1.a.shape[0]), (0, g_2.a.shape[1]-g_1.a.shape[1])), mode='constant', constant_values=0)
            a = _a + g_2.a
        #x
        if not (g_1.x.shape > g_2.x.shape or g_1.x.shape < g_2.x.shape):
            x = g_1.x + g_2.x
        elif (g_1.x.shape > g_2.x.shape):
            _x = np.pad(g_2.x, ((0, g_1.x.shape[0]-g_2.x.shape[0]), (0, g_1.x.shape[1]-g_2.x.shape[1])), mode='constant', constant_values=0)
            x = g_1.x + _x
        elif (g_1.x.shape < g_2.x.shape):
            _x = np.pad(g_1.x, ((0, g_2.x.shape[0]-g_1.x.shape[0]), (0, g_2.x.shape[1]-g_1.x.shape[1])), mode='constant', constant_values=0)
            x = _x + g_2.x
        #e
        if not (g_1.e.shape > g_2.e.shape or g_1.e.shape < g_2.e.shape):
            e = g_1.e + g_2.e
        elif (g_1.e.shape > g_2.e.shape):
            _e = np.pad(g_2.e, ((0, g_1.e.shape[0]-g_2.e.shape[0]), (0, g_1.e.shape[1]-g_2.e.shape[1])), mode='constant', constant_values=0)
            e = g_1.e + _e
        elif (g_1.e.shape < g_2.e.shape):
            _e = np.pad(g_1.e, ((0, g_2.e.shape[0]-g_1.e.shape[0]), (0, g_2.e.shape[1]-g_1.e.shape[1])), mode='constant', constant_values=0)
            e = _e + g_2.e
        g_n = Graph()
        g_n.a = a
        g_n.x = x
        g_n.e = e
        graphs.append(g_n)

    return graphs

def merge_mean(idx_a, idx_b, dataset, config):
    # mode=="type-merge-mean"
    assert(len(idx_a)==len(idx_b))
    graphs = []
    for i in range(len(idx_a)):
        g_1, g_2 = dataset[idx_a[i]], dataset[idx_b[i]]
        assert(g_1.x.shape[1]==config['x_shape1'] and g_2.x.shape[1]==config['x_shape1'])
        assert(g_2.e.shape[1]==3 and g_2.e.shape[1]==3)
        #a
        if not (g_1.a.shape > g_2.a.shape or g_1.a.shape < g_2.a.shape):
            a = (g_1.a + g_2.a) // 2
        elif (g_1.a.shape > g_2.a.shape):
            _a = np.pad(g_2.a, ((0, g_1.a.shape[0]-g_2.a.shape[0]), (0, g_1.a.shape[1]-g_2.a.shape[1])), mode='constant', constant_values=0)
            a = (g_1.a + _a) // 2
        elif (g_1.a.shape < g_2.a.shape):
            _a = np.pad(g_1.a, ((0, g_2.a.shape[0]-g_1.a.shape[0]), (0, g_2.a.shape[1]-g_1.a.shape[1])), mode='constant', constant_values=0)
            a = (_a + g_2.a) // 2
        #x
        if not (g_1.x.shape > g_2.x.shape or g_1.x.shape < g_2.x.shape):
            x = (g_1.x + g_2.x) // 2
        elif (g_1.x.shape > g_2.x.shape):
            _x = np.pad(g_2.x, ((0, g_1.x.shape[0]-g_2.x.shape[0]), (0, g_1.x.shape[1]-g_2.x.shape[1])), mode='constant', constant_values=0)
            x = (g_1.x + _x) // 2
        elif (g_1.x.shape < g_2.x.shape):
            _x = np.pad(g_1.x, ((0, g_2.x.shape[0]-g_1.x.shape[0]), (0, g_2.x.shape[1]-g_1.x.shape[1])), mode='constant', constant_values=0)
            x = (_x + g_2.x) // 2
        #e
        if not (g_1.e.shape > g_2.e.shape or g_1.e.shape < g_2.e.shape):
            e = (g_1.e + g_2.e) // 2
        elif (g_1.e.shape > g_2.e.shape):
            _e = np.pad(g_2.e, ((0, g_1.e.shape[0]-g_2.e.shape[0]), (0, g_1.e.shape[1]-g_2.e.shape[1])), mode='constant', constant_values=0)
            e = (g_1.e + _e) // 2
        elif (g_1.e.shape < g_2.e.shape):
            _e = np.pad(g_1.e, ((0, g_2.e.shape[0]-g_1.e.shape[0]), (0, g_2.e.shape[1]-g_1.e.shape[1])), mode='constant', constant_values=0)
            e = (_e + g_2.e) // 2
        g_n = Graph()
        g_n.a = a
        g_n.x = x
        g_n.e = e
        graphs.append(g_n)

    return graphs