import numpy as np
from spektral.datasets import TUDataset, QM9

from data.ogb_helper import ogb_available_datasets, OGBDataset

from itertools import combinations

from scipy.stats import rankdata

def _load_data(config):
    '''
    Loads a dataset from [TUDataset, OGB]
    '''
    name = config['dataset']
    # if name == 'QM9':
    #     dataset = QM9(amount=10)# 1000 and 100000 ok
    if name in TUDataset.available_datasets():
        dataset = TUDataset(name)
        config['x_shape1'] = 28
    elif name in ogb_available_datasets():
        dataset= OGBDataset(name)
        config['x_shape1'] = 9
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
    ys = [g.y for g in data]

    return rankdata(ys, method='dense')

def sample_preference_pairs(graphs):
    c = [(a, b, check_util(graphs, a,b)) for a, b in combinations(range(len(graphs)), 2)]
    idx_a = []
    idx_b = []
    target = []
    for id_a, id_b, t in c:
        idx_a.append(id_a)
        idx_b.append(id_b)
        target.append(t)
    return np.array(idx_a), np.array(idx_b), np.array(target).reshape(-1, 1)

def check_util(data, index_a, index_b):
    a = data[index_a]
    b = data[index_b]
    util_a = a.y
    util_b = b.y
    if util_a >= util_b:
        return 1
    else:
        return 0


def get_data(config):
    seed = config['seed']
    train_test_split = config['train_test_split']

    # Load data
    data, config['n_out'] = _load_data(config)
    # Split data
    train_data, test_data = _split_data(data, train_test_split, seed)
    print("len test_data:", len(test_data))

    ground_truth_ranking = _rankData(test_data)

    return train_data, test_data, ground_truth_ranking