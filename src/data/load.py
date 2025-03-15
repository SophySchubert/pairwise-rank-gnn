from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import TUDataset
import numpy as np

from data.misc import sample_preference_pairs, rank_data, transform_dataset_to_pair_dataset, transform_dataset_to_pair_dataset_torch

def _ogb_available_datasets():
    return ['ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-mollipo']

def _tud_available_datasets():
    return ['aspirin', 'ZINC_full']

def _load_data(config):
    '''
    Loads a dataset from [TUDataset, OGB]
    '''
    name = config['dataset']
    if name in _ogb_available_datasets():
        dataset= PygGraphPropPredDataset(name=name)
        config['num_node_features'] = dataset.num_node_features
    # elif name in _tud_available_datasets():
    #     dataset = TUDataset(root='/dataset/'+name, name=name)
    #     if not hasattr(dataset, 'get_idx_split'):
    #         VALID_SPLIT = 0.8
    #         TEST_SPLIT = 0.1
    #         train_size = int(VALID_SPLIT * len(dataset))
    #         valid_size = int(TEST_SPLIT * len(dataset))
    #         test_size = len(dataset) - train_size - valid_size
    #         # Split the dataset
    #         train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    else:
        raise ValueError(f'Dataset {name} unknown')

    return dataset, dataset.get_idx_split()

def get_data(config):
    dataset, split_idx = _load_data(config)
    # Split the dataset into training, validation, and test sets
    train_dataset = dataset[split_idx['train']]
    valid_dataset = dataset[split_idx['valid']]
    test_dataset = dataset[split_idx['test']]

    train_prefs = sample_preference_pairs(train_dataset)
    valid_prefs = sample_preference_pairs(valid_dataset)
    test_ranking = rank_data([g.y.item() for g in test_dataset])

    # create pairs and targets
    if config['mode'] == 'default':
        _tmp = range(len(test_dataset))
        test_prefs = np.array(list(zip(_tmp, _tmp, _tmp))) # differs due to only needed for prediction
        return train_dataset, valid_dataset, test_dataset, train_prefs, valid_prefs, test_prefs, test_ranking
    elif config['mode'] == 'fc_weight':
        train_dataset = transform_dataset_to_pair_dataset(train_dataset, train_prefs, config)
        valid_dataset = transform_dataset_to_pair_dataset(valid_dataset, valid_prefs, config)
        test_prefs = sample_preference_pairs(test_dataset)
        test_dataset = transform_dataset_to_pair_dataset(test_dataset, test_prefs, config)
        return train_dataset, valid_dataset, test_dataset, test_prefs, test_ranking
    elif config['mode'] == 'fc_extra':
        train_dataset = transform_dataset_to_pair_dataset_torch(train_dataset, train_prefs, config)
        valid_dataset = transform_dataset_to_pair_dataset_torch(valid_dataset, valid_prefs, config)
        test_prefs = sample_preference_pairs(test_dataset)
        test_dataset = transform_dataset_to_pair_dataset_torch(test_dataset, test_prefs, config)
        return train_dataset, valid_dataset, test_dataset, test_prefs, test_ranking