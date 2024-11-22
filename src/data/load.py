from spektral.data import DisjointLoader
from spektral.datasets import TUDataset
import numpy as np


def _load_data(name: str):
    '''
    Loads a dataset from TUDataset
    TODO: Expand to include more datasets
    '''
    dataset = TUDataset(name)
    return dataset

def split_data(config):
    np.random.seed(config['seed'])

    data = _load_data(config['dataset'])
    np.random.shuffle(data)

    split = int(config['train_test_split['] * len(data))
    data_train, data_test = data[:split], data[split:]
    n_labels = data_train.n_labels

    loader_train = DisjointLoader(data_train, batch_size=config['batch_size'], epochs=config['epochs'])
    loader_test = DisjointLoader(data_test, batch_size=config['batch_size'])

    return loader_train, loader_test, n_labels

