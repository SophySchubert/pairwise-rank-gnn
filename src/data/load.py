import numpy as np
from spektral.datasets import TUDataset
from spektral.data import DisjointLoader
# from ogb.graphproppred import PygGraphPropPredDataset

def _load_data(name: str):
    '''
    Loads a dataset from [TUDataset, OGB]
    TODO: Add more datasets
    '''

    if name in TUDataset.available_datasets():
        return TUDataset(name)
    elif name in _obg_available_datasets():
        pass
        # return PygGraphPropPredDataset(name=name, root='data/')
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

    loader_train = DisjointLoader(train_data, batch_size=config['batch_size'], epochs=config['epochs'])
    loader_test = DisjointLoader(test_data, batch_size=config['batch_size'])

    return loader_train, loader_test, n_labels


def _obg_available_datasets():
    return ['ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molmuv', 'ogbg-molpcba', 'ogbg-molsider', 'ogbg-moltox21', 'ogbg-moltoxcast', 'ogbg-molhiv', 'ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-mollipo', 'ogbg-molchembl', 'ogbg-ppa', 'ogbg-code2']
