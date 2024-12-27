import networkx as nx
import numpy as np
import tensorflow as tf
from spektral.data import DisjointLoader
from spektral.data.loaders import tf_loader_available
import scipy.sparse as sp
from spektral.data.utils import (
    collate_labels_disjoint,
    prepend_none,
    sp_matrices_to_sp_tensors,
    to_disjoint
)

def spektral_graph_to_nx_graph(spektral_graph):
    # Create a NetworkX graph from the adjacency matrix
    g = nx.from_scipy_sparse_matrix(spektral_graph.a)

    # Add node features to the NetworkX graph
    for i, features in enumerate(spektral_graph.x):
        g.nodes[i]['features'] = features

    return g


def to_tf_signature(signature):
    """
    Converts a Dataset signature to a TensorFlow signature. Extended keys (idx_a, idx_b) for MyDisjointLoader.
    :param signature: a Dataset signature.
    :return: a TensorFlow signature.
    """
    output = []
    keys = ["x", "a", "e", "i", "idx_a", "idx_b"]
    for k in keys:
        if k in signature:
            shape = signature[k]["shape"]
            dtype = signature[k]["dtype"]
            spec = signature[k]["spec"]
            output.append(spec(shape, dtype))
    output = tuple(output)
    if "y" in signature:
        shape = signature["y"]["shape"]
        dtype = signature["y"]["dtype"]
        spec = signature["y"]["spec"]
        output = (output, spec(shape, dtype))

    return output

class MyDisjointLoader(DisjointLoader):
    """
    Extension of DisjointLoader class from spektral library. Additionally to data and targets, it also returns ranking pair indices.
    A Loader for [disjoint mode](https://graphneural.network/data-modes/#disjoint-mode).

    This loader represents a batch of graphs via their disjoint union.

    The loader automatically computes a batch index tensor, containing integer
    indices that map each node to its corresponding graph in the batch.

    The adjacency matrix os returned as a SparseTensor, regardless of the input.

    If `node_level=False`, the labels are interpreted as graph-level labels and
    are stacked along an additional dimension.
    If `node_level=True`, then the labels are stacked vertically.

    **Note:** TensorFlow 2.4 or above is required to use this Loader's `load()`
    method in a Keras training loop.

    **Arguments**

    - `dataset`: a graph Dataset;
    - `node_level`: bool, if `True` stack the labels vertically for node-level
    prediction;
    - `batch_size`: size of the mini-batches;
    - `epochs`: number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: whether to shuffle the data at the start of each epoch.

    **Output**

    For each batch, returns a tuple `(inputs, labels)`.

    `inputs` is a tuple containing:

    - `x`: node attributes of shape `[n_nodes, n_node_features]`;
    - `a`: adjacency matrices of shape `[n_nodes, n_nodes]`;
    - `e`: edge attributes of shape `[n_edges, n_edge_features]`;
    - `i`: batch index of shape `[n_nodes]`.

    `labels` have shape `[batch, n_labels]` if `node_level=False` or
    `[n_nodes, n_labels]` otherwise.

    """

    def __init__(
        self, dataset, node_level=False, batch_size=1, epochs=None, shuffle=True, seed=42
    ):
        self.node_level = node_level
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)
        self.seed = seed

    def collate(self, batch):
        idx_a, idx_b, target = self.sample_preference_pairs(batch, seed=self.seed)
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_disjoint(y, node_level=self.node_level)

        output = to_disjoint(**packed)
        output = sp_matrices_to_sp_tensors(output)

        if len(output) == 1:
            output = output[0]

        if y is None:
            return output
        else:
            return (output, idx_a, idx_b), target

    def load(self):
        print("load")
        if not tf_loader_available:
            raise RuntimeError(
                "Calling DisjointLoader.load() requires " "TensorFlow 2.4 or greater."
            )
        return tf.data.Dataset.from_generator(
            lambda: self, output_signature=self.tf_signature()
        )

    def tf_signature(self):
        """
        Adjacency matrix has shape [n_nodes, n_nodes]
        Node features have shape [n_nodes, n_node_features]
        Edge features have shape [n_edges, n_edge_features]
        Targets have shape [*, n_labels]
        Pairs have shape [*, 2]
        """
        signature = self.dataset.signature
        if "y" in signature:
            signature["y"]["shape"] = prepend_none(signature["y"]["shape"]) #(12800,) #(None, 1)
        if "a" in signature:
            signature["a"]["spec"] = tf.SparseTensorSpec

        signature["i"] = dict()
        signature["i"]["spec"] = tf.TensorSpec
        signature["i"]["shape"] = (None,)
        signature["i"]["dtype"] = tf.as_dtype(tf.int64)

        signature["idx_a"] = dict()
        signature["idx_a"]["spec"] = tf.TensorSpec
        signature["idx_a"]["shape"] = (None,)
        signature["idx_a"]["dtype"] = tf.as_dtype(tf.int64)
        signature["idx_b"] = dict()
        signature["idx_b"]["spec"] = tf.TensorSpec
        signature["idx_b"]["shape"] = (None,)
        signature["idx_b"]["dtype"] = tf.as_dtype(tf.int64)

        return to_tf_signature(signature)

    def sample_preference_pairs(self, graphs, radius=4, sampling_ratio=100, seed=42):
        size = len(graphs)
        sample_size = size * radius * sampling_ratio
        r = np.arange(size)
        S = sp.csr_matrix((r, (r, r)), shape=(size, size))
        parts = np.split(S.data, S.indptr[1:-1])
        rnd = np.random.default_rng(seed)
        for part in parts:
            rnd.shuffle(part)
        idx_a = np.empty((sample_size,), dtype=np.int64)
        idx_b = np.empty((sample_size,), dtype=np.int64)
        target = np.ones((sample_size,), dtype=np.float64)
        k = 0
        for i in range(size):
            part = parts[i]
            psize = len(part)
            for d in range(radius):
                ni = (i + d + 1) % size
                npart = parts[ni]
                npsize = len(npart)
                for j in range(sampling_ratio):
                    npart_offset = np.roll(npart, d * sampling_ratio + j)
                    idx_a[k:k + psize] = part
                    if npsize < psize:
                        idx_b[k:k + npsize] = npart_offset
                        idx_b[k + npsize:k + psize] = npart_offset[:psize - npsize]
                    else:
                        idx_b[k:k + psize] = npart_offset[:psize]
                    if ni < i:
                        target[k:k + psize] = 0
                    k += psize
        return idx_a, idx_b, target