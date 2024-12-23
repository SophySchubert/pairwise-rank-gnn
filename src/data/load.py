import numpy as np
from spektral.datasets import TUDataset, QM9

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

    # ids = np.array(ids, dtype=np.int64)
    # targets = np.array(targets, dtype=np.int64)
    # print(ids.dtype)
    # print(targets.dtype)

    for i, tuple in enumerate(ids):
        dataset[tuple[0]].pair = tuple
        dataset[tuple[0]].pair_target = targets[i]
        dataset[tuple[1]].pair = tuple
        dataset[tuple[1]].pair_target = targets[i]

    return ids, targets

def get_data(config):
    seed = config['seed']
    train_test_split = config['train_test_split']
    name = config['dataset']
    pairwise = config['pairwise']

    # Load data
    data, config['n_out'] = _load_data(name)
    # Split data
    train_data, test_data = _split_data(data, train_test_split, seed)
    # Create pairs
    train_pairs, train_targets = None, None,
    test_pairs, test_targets = None, None
    if pairwise:
        train_pairs, train_targets = _sample_pairs(train_data)
        test_pairs, test_targets = _sample_pairs(test_data)

    return train_data, train_pairs, train_targets, test_data, test_pairs, test_targets