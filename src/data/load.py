import numpy as np

from spektral.datasets import TUDataset, QM9
from data.ogb_helper import OGBDataset, ogb_available_datasets


def _load_data(name: str):
    '''
    Loads a dataset from [TUDataset, OGB]
    '''

    if name == 'QM9':
        dataset = QM9(amount=100000)# 1000 and 100000 ok
    elif name in TUDataset.available_datasets():
        dataset = TUDataset(name)
    elif name in ogb_available_datasets():
        dataset = OGBDataset(name)
    else:
        raise ValueError(f'Dataset {name} unknown')

    return dataset, dataset.n_labels

def _sample_pairs(dataset):
    '''
    Sample pairs of graphs from a dataset
    '''
    assert(len(dataset) % 2 == 0)

    _pair_a = dataset[::2]
    _pair_b = dataset[1::2]

    return tuple(zip(_pair_a, _pair_b))

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
        train_data = _sample_pairs(train_data)
        test_data = _sample_pairs(test_data)

    return train_data, test_data





if __name__ == "__main__":
    pass





















































