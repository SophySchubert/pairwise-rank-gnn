from spektral.data import DisjointLoader
from spektral.datasets import TUDataset
import numpy as np


def _load_data(name: str):
    '''
    Loads a dataset from TUDataset
    TODO: Expand to include more datasets
    '''
    if name in TUDataset.available_datasets():
        return TUDataset(name)
    # elif name in []:
    #     return None
    else:
        raise ValueError(f'Dataset {name} unknown')


def split_data(config):
    # TODO: Split into two training points/graphs
    np.random.seed(config['seed'])

    data = _load_data(config['dataset'])
    np.random.shuffle(data)

    # Split the dataset into train and test sets
    train_size = int(len(data) * config['train_test_split'])
    train_data = data[:train_size]
    test_data = data[train_size:]
    n_labels = train_data.n_labels

    loader_train = DisjointLoader(train_data, batch_size=config['batch_size'])
    loader_test = DisjointLoader(test_data, batch_size=config['batch_size'])

    return loader_train, loader_test, n_labels

