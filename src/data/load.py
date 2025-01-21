import numpy as np
from spektral.datasets import TUDataset, QM9

from data.ogb_helper import ogb_available_datasets, OGBDataset


def _load_data(name: str):
    '''
    Loads a dataset from [TUDataset, OGB]
    '''
    # if name == 'QM9':
    #     dataset = QM9(amount=10)# 1000 and 100000 ok
    if name in TUDataset.available_datasets():
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

def _rankData(data):
    indexed_graphs= list(enumerate(data))

    sorted_indexed_graphs = sorted(indexed_graphs, key=lambda x: x[1].y)

    sorted_graphs = [g for index, g in sorted_indexed_graphs]
    original_indices = [index for index, g in sorted_indexed_graphs]

    return original_indices#zip(sorted_graphs, original_indices)


def get_data(config):
    seed = config['seed']
    train_test_split = config['train_test_split']
    name = config['dataset']

    # Load data
    data, config['n_out'] = _load_data(name)
    ground_truth_ranking = _rankData(data)
    # Split data
    train_data, test_data = _split_data(data, train_test_split, seed)

    return train_data, test_data, ground_truth_ranking