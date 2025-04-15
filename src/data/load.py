import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import ZINC
import numpy as np

from data.misc import sample_preference_pairs, rank_data, transform_dataset_to_pair_dataset, transform_dataset_to_pair_dataset_torch

def _ogb_available_datasets():
    return ['ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-mollipo']

def _load_data(config):
    '''
    Loads a dataset from [pyg, OGBG]
    '''
    name = config['dataset']
    if name in _ogb_available_datasets():
        dataset = PygGraphPropPredDataset(name=name)
        config['num_node_features'] = dataset.num_node_features
        config['max_num_nodes'] = max([d.num_nodes for d in dataset])
    elif name == 'ZINC':
        dataset = ZINC(root='./data/ZINC', subset=True)
        config['num_node_features'] = dataset.num_node_features
        config['max_num_nodes'] = max([d.num_nodes for d in dataset])
    else:
        raise ValueError(f'Dataset {name} unknown')

    return dataset, dataset.get_idx_split()

def get_data(config):
    dataset, split_idx = _load_data(config)
    # Split the dataset into training, validation, and test sets
    VALID_SPLIT = 0.8
    TEST_SPLIT = 0.1
    train_size = int(VALID_SPLIT * len(dataset))
    valid_size = int(TEST_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size, valid_size, test_size])
    # train_dataset = dataset[split_idx['train']]
    # valid_dataset = dataset[split_idx['test']]
    # test_dataset = dataset[split_idx['valid']]

    train_prefs = sample_preference_pairs(train_dataset)
    valid_prefs = sample_preference_pairs(valid_dataset)
    test_prefs = sample_preference_pairs(test_dataset)
    test_ranking = rank_data([g.y.item() for g in test_dataset])

    # create pairs and targets
    if config['mode'] == 'default':
        test_prefs = np.array([[0, i, 0] for i in range(0, len(test_dataset))])# differs due to only needed for prediction
        return train_dataset, valid_dataset, test_dataset, train_prefs, valid_prefs, test_prefs, test_ranking
    elif config['mode'] == 'gat_attention' or config['mode'] == 'nagsl_attention':
        return train_dataset, valid_dataset, test_dataset, train_prefs, valid_prefs, test_prefs, test_ranking
    elif config['mode'] == 'fc_weight' or config['mode'] == 'my_attention':
        train_dataset = transform_dataset_to_pair_dataset(train_dataset, train_prefs, config)
        valid_dataset = transform_dataset_to_pair_dataset(valid_dataset, valid_prefs, config)
        test_dataset = transform_dataset_to_pair_dataset(test_dataset, test_prefs, config)
        return train_dataset, valid_dataset, test_dataset, test_prefs, test_ranking
    elif config['mode'] == 'fc_extra':
        train_dataset = transform_dataset_to_pair_dataset_torch(train_dataset, train_prefs, config)
        valid_dataset = transform_dataset_to_pair_dataset_torch(valid_dataset, valid_prefs, config)
        test_dataset = transform_dataset_to_pair_dataset_torch(test_dataset, test_prefs, config)
        return train_dataset, valid_dataset, test_dataset, test_prefs, test_ranking
    else:
        raise ValueError(f'Unknown mode {config["mode"]}')