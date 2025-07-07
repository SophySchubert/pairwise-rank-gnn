import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import numpy as np
from data.misc import nagsl_pair_attention_transform, transform_dataset_to_pair_dataset_torch

class CustomDataLoader(DataLoader):
    """
    Custom DataLoader for pairwise graph data used in this framework.
    Based on PyTorch Geometric's DataLoader.
    The default attribute for the dataset is used for the pairs to be sampled in each batch.
    Entire dataset is stored in the entire_dataset attribute.
    """
    def __init__(self, pairs_and_targets, dataset, batch_size=1, shuffle=False, mode='default', config=None, **kwargs):
        super().__init__(pairs_and_targets, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.entire_dataset = dataset
        self.mode = mode
        self.config = config

    def __iter__(self):
        for batch in super().__iter__():
            batch = self.augment_batch(batch)
            yield batch

    def augment_batch(self, batch):
        '''
        Take each batch and load required data from the entire dataset
        Batch is a list of pairs and targets
        '''

        idx_a, idx_b, target = zip(*[(x[0], x[1], x[2]) for x in batch])
        if self.mode == 'default':
            data = self.get_data_from_indices(idx_a, idx_b, unique=True) # retrieve data from dataset
            idx_a, idx_b = self.reindex_ids(idx_a, idx_b) # reindex ids to not get a out of bounds error in the NN

            # Create a DataBatch object
            data_batch = Batch.from_data_list(data)

            # Add idx_a and idx_b to the DataBatch object
            data_batch.idx_a = torch.tensor(idx_a)
            data_batch.idx_b = torch.tensor(idx_b)
            data_batch.y = torch.tensor(target)

        elif self.mode == 'gat_attention':
            data_a = self.get_data_from_indices(idx_a, [], unique=False)
            data_b = self.get_data_from_indices(idx_b, [], unique=False)

            data_a = Batch.from_data_list(data_a)
            data_b = Batch.from_data_list(data_b)
            data_a.y = torch.tensor(target)
            data_batch = [data_a, data_b] # this mode uses a different way to get the pairwise data

        elif self.mode == 'nagsl_attention':
            # NAGSL framework
            data_a = self.get_data_from_indices(idx_a, [], unique=False)
            data_b = self.get_data_from_indices(idx_b, [], unique=False)

            data_a = Batch.from_data_list(data_a)
            data_b = Batch.from_data_list(data_b)
            data_batch = nagsl_pair_attention_transform((data_a, data_b), torch.tensor(target), self.config)
        elif self.mode == 'rank_mask':
            combined, solo = transform_dataset_to_pair_dataset_torch(dataset=self.entire_dataset, prefs=batch, config=self.config, from_loader=True)
            data_batch = Batch.from_data_list(combined)
            data_batch_solo = Batch.from_data_list(solo)

            data_batch.y = torch.tensor(target)
            data_batch.solo_edge_index = data_batch_solo.edge_index
            data_batch.solo_batch = data_batch_solo.batch
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return data_batch

    def get_data_from_indices(self, idx_a, idx_b, unique=False):
        '''
        Return a sub dataset from the ids of the batch
        '''
        if unique:
            # Some methods do not need duplicates, pairs then get referenced by ids
            ids = np.unique(np.concatenate((idx_a, idx_b)))
        else:
            ids = idx_a

        required_data = [self.entire_dataset[int(i)] for i in ids]
        return required_data

    def reindex_ids(self, idx_a, idx_b):
        '''
        Transforms the ids of a batch to be of in the same range as the batch size
        Before reindexing ids reference the entire dataset afterwards they reference the batch.
        '''
        # idx_a = idx_a.numpy()
        # idx_b = idx_b.numpy()
        ids = np.unique(np.concatenate((idx_a, idx_b)))

        # Create a mapping from unique elements to the range [0, length)
        mapping = {element: idx for idx, element in enumerate(ids)}

        # Apply the mapping to both arrays
        mapped_a = np.array([mapping[element.item()] for element in idx_a])
        mapped_b = np.array([mapping[element.item()] for element in idx_b])

        return mapped_a, mapped_b