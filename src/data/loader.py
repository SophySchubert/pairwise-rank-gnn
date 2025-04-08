import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import numpy as np


class CustomDataLoader(DataLoader):
    def __init__(self, pairs_and_targets, dataset, batch_size=1, shuffle=False, attention=False, **kwargs):
        super().__init__(pairs_and_targets, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.entire_dataset = dataset
        self.attention = attention

    def __iter__(self):
        for batch in super().__iter__():
            batch = self.augment_batch(batch)
            yield batch

    def augment_batch(self, batch):
        # batch is a list of pairs and targets
        idx_a, idx_b, target = zip(*[(x[0], x[1], x[2]) for x in batch])
        if not self.attention:
            data = self.get_data_from_indices(idx_a, idx_b)
            idx_a, idx_b = self.reindex_ids(idx_a, idx_b)

            # Create a DataBatch object
            data_batch = Batch.from_data_list(data)

            # Add idx_a and idx_b to the DataBatch object
            data_batch.idx_a = torch.tensor(idx_a)
            data_batch.idx_b = torch.tensor(idx_b)
            data_batch.y = torch.tensor(target)
        else:
            data_a = self.get_data_from_indices(idx_a, [], attention=True)
            data_b = self.get_data_from_indices(idx_b, [], attention=True)

            data_a = Batch.from_data_list(data_a)
            data_b = Batch.from_data_list(data_b)
            data_a.y = torch.tensor(target)
            data_batch = [data_a, data_b]

        return data_batch

    def get_data_from_indices(self, idx_a, idx_b, attention=False):
        ids = np.unique(np.concatenate((idx_a, idx_b)))
        if attention:
            ids = idx_a
        required_data = [self.entire_dataset[int(i)] for i in ids]
        return required_data

    def reindex_ids(self, idx_a, idx_b):
        ids = np.unique(np.concatenate((idx_a, idx_b)))
        idx_a = np.array(idx_a)
        idx_b = np.array(idx_b)

        # Create a mapping from unique elements to the range [0, length)
        mapping = {element: idx for idx, element in enumerate(ids)}

        # Apply the mapping to both arrays
        mapped_a = np.array([mapping[element] for element in idx_a])
        mapped_b = np.array([mapping[element] for element in idx_b])

        return mapped_a, mapped_b