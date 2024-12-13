import numpy as np
import tensorflow as tf
from spektral.datasets import TUDataset, QM9
from data.ogb_helper import OGBDataset, ogb_available_datasets


def _load_data(name: str):
    '''
    Loads a dataset from [TUDataset, OGB]
    '''

    if name == 'QM9':
        dataset = QM9(amount=1000)# 1000 and 100000 ok
    elif name in TUDataset.available_datasets():
        dataset = TUDataset(name)
    elif name in ogb_available_datasets():
        dataset = OGBDataset(name)
    else:
        raise ValueError(f'Dataset {name} unknown')

    return dataset, dataset.n_labels

def _split_data(data, train_test_split, seed):
    '''
    Split the data into train and test sets
    https://github.com/KIuML/PLR_SS22/blob/master/exercise_07.ipynb ?
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
    dataset = [d.y for d in dataset]
    _pairs_even = dataset[::2]
    _pairs_odd = dataset[1::2]
    _pairs_even_ids = range(0, len_dataset, 2)
    _pairs_odd_ids = range(1, len_dataset, 2)

    data = list(zip(_pairs_even, _pairs_odd))
    ids = list(zip(_pairs_even_ids, _pairs_odd_ids))
    targets = [max(d) for d in data]

    return ids, targets

def get_data(config):
    seed = config['seed']
    train_test_split = config['train_test_split']
    name = config['dataset']
    pairwise = config['pairwise']

    # Load data
    data, n_out = _load_data(name)
    config['n_out'] = n_out
    # Split data
    train_data, test_data = _split_data(data, train_test_split, seed)
    # Create pairs
    if pairwise:
        train_pairs, train_targets = _sample_pairs(train_data)
        test_pairs, test_targets = _sample_pairs(test_data)

    return train_data, train_pairs, train_targets, test_data, test_pairs, test_targets