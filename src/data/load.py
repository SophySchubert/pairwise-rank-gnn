import numpy as np
from spektral.datasets import TUDataset, QM9
import scipy.sparse as sp

from data.ogb_helper import ogb_available_datasets, OGBDataset


def _load_data(name: str):
    '''
    Loads a dataset from [TUDataset, OGB]
    '''
    if name == 'QM9':
        dataset = QM9(amount=1000)# 1000 and 100000 ok
    elif name in TUDataset.available_datasets():
        dataset = TUDataset(name)
    elif name in ogb_available_datasets():
        dataset= OGBDataset(name)
    else:
        raise ValueError(f'Dataset {name} unknown')

    return dataset, dataset.n_labels

def _split_data(data, train_test_split, seed):
    '''
    Split the data into train and test sets
    '''
    np.random.seed(seed)
    idxs = np.random.permutation(len(data))
    split = int(train_test_split * len(data))
    idx_train, idx_test = np.split(idxs, [split])
    train, test = data[idx_train], data[idx_test]
    return train, test

def _sample_pairs(dataset):
    '''
    Sample pairs of graphs from a dataset
    '''
    assert(len(dataset) % 2 == 0)
    len_dataset = len(dataset)
    dataset_targets = [d.y for d in dataset]
    _pairs_even = dataset_targets[::2]
    _pairs_odd = dataset_targets[1::2]
    _pairs_even_ids = range(0, len_dataset, 2)
    _pairs_odd_ids = range(1, len_dataset, 2)

    data = list(zip(_pairs_even, _pairs_odd))
    ids = np.array(list(zip(_pairs_even_ids, _pairs_odd_ids)))
    targets = [np.maximum(d[0], d[1]) for d in data]

    for i, tuple in enumerate(ids):
        dataset[tuple[0]].pair = tuple
        dataset[tuple[0]].pair_target = targets[i]
        dataset[tuple[1]].pair = tuple
        dataset[tuple[1]].pair_target = targets[i]

    return ids, targets

def sample_preference_pairs(graphs, radius=4, sampling_ratio=100, seed=42):
        np.random.seed(seed)
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
        return idx_a, idx_b, target.reshape(-1, 1)

def get_data(config):
    seed = config['seed']
    train_test_split = config['train_test_split']
    name = config['dataset']

    # Load data
    data, config['n_out'] = _load_data(name)
#     max_val = max(data, key=lambda x: x.y)
#     min_val = min(data, key=lambda x: x.y)
#     print(max_val.y, min_val.y)#wie ich die ranken soll und dann kann ich bestimmen wie ich target bilde
#     different=[]
#     for d in data:
#         different.append(d.y[0])
#     print(len(data), len(set(different)))
    config['max_nodes'] = max(g.n_nodes for g in data)
    # Split data
    train_data, test_data = _split_data(data, train_test_split, seed)
    # Sample pairs
    train_idx_a, train_idx_b, train_tragets = sample_preference_pairs(train_data, 4, 100, seed)
    test_idx_a, test_idx_b, test_targets = sample_preference_pairs(test_data, 4, 100, seed)

    return [train_data, (train_idx_a, train_idx_b), train_tragets], [test_data, (test_idx_a, test_idx_b), test_targets]