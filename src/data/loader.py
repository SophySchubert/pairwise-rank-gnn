import numpy as np
import tensorflow as tf
from spektral.data import DisjointLoader
from spektral.data.loaders import tf_loader_available
from spektral.data.utils import (
    prepend_none,
    sp_matrices_to_sp_tensors,
    to_disjoint,
    collate_labels_disjoint,
    batch_generator
)
from itertools import combinations
from sklearn.utils import shuffle as sk_shuffle


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
    """

    def __init__(
            self, dataset, node_level=False, batch_size=1, epochs=None, shuffle=True, seed=42
    ):
        self.node_level = node_level
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)
        self.seed = seed

    def collate(self, batch):
        idx_a, idx_b, target = self.sample_preference_pairs(batch)
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_disjoint(y, node_level=self.node_level)

        output = to_disjoint(**packed)
        output = sp_matrices_to_sp_tensors(output)

        return output + (idx_a, idx_b), target

    def load(self):
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

    def sample_preference_pairs(self, graphs):
        c = [(a, b, self.check_util(graphs, a,b)) for a, b in combinations(range(len(graphs)), 2)]
        idx_a = []
        idx_b = []
        target = []
        for id_a, id_b, t in c:
            idx_a.append(id_a)
            idx_b.append(id_b)
            target.append(t)
        return np.array(idx_a), np.array(idx_b), np.array(target).reshape(-1, 1)

    def check_util(self, data, index_a, index_b):
        a = data[index_a]
        b = data[index_b]
        util_a = a.y
        util_b = b.y
        if util_a >= util_b:
            return 1
        else:
            return 0

class CustomDataLoader(tf.keras.utils.Sequence):
    def __init__(self, data, pairs, targets, batch_size=32, shuffle=True, seed=42):
        print("CustomDataLoader.__init__")
        self.data = data
        self.pairs = pairs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indices = np.arange(len(self.pairs))
        self.node_level = False
        self.on_epoch_end()

    def __len__(self):
        print("CustomDataLoader.__len__")
        return int(np.floor(len(self.pairs) / self.batch_size))

    def __getitem__(self, index):
        print("CustomDataLoader.__getitem__")
        indices = np.array(self.indices[index*self.batch_size:(index+1)*self.batch_size])
        batch_pairs = self.pairs[indices]
        batch_data, idx_a, idx_b = self.get_batch_data(batch_pairs)
        batch_targets = self.targets[indices]
        #self.info(batch_pairs, batch_data, batch_targets)

        #disjointLoader content start
        packed = self.pack(batch_data)
        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_disjoint(y, node_level=self.node_level)
        output = to_disjoint(**packed)
        output = sp_matrices_to_sp_tensors(output)
        #disjointLoader content end
        output = output + (idx_a, idx_b)
        print(len(output))
        print(output)

        return output, batch_targets #mit yield den Fehler:ValueError: not enough values to unpack (expected 6, got 1)

    def on_epoch_end(self):
        print("CustomDataLoader.on_epoch_end")
        if self.shuffle:
            np.random.shuffle(self.indices)
            self.pairs, self.targets = sk_shuffle(self.pairs, self.targets)

    def get_batch_data(self, pairs):
        print("CustomDataLoader.get_batch_data")
        p1, p2 = zip(*[(x[0], x[1]) for x in pairs])
        required_indices = np.array(list(set(p1+p2)))
        required_data = self.data[required_indices]
        return required_data, p1, p2

    def pack(self, batch):
        print("CustomDataLoader.pack")
        """
        Given a batch of graphs, groups their attributes into separate lists and packs
        them in a dictionary.

        For instance, if a batch has three graphs g1, g2 and g3 with node
        features (x1, x2, x3) and adjacency matrices (a1, a2, a3), this method
        will return a dictionary:

        ```python
        >>> {'a_list': [a1, a2, a3], 'x_list': [x1, x2, x3]}
        ```

        :param batch: a list of `Graph` objects.
        """
        output = [list(elem) for elem in zip(*[g.numpy() for g in batch])]
        keys = [k + "_list" for k in self.data.signature.keys()]
        return dict(zip(keys, output))

    def info(self, b_p, b_d, b_t):
        print("CustomDataLoader.info")
        print(f"len pairs:{len(self.pairs)}")
        print(f"len indices:{len(self.pairs)}")
        print(f"batch_pairs:{b_p}")
        print(f"batch_data:{b_d}")
        print(f"batch_target:{b_t}")


