import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_torch_sparse_tensor
import numpy as np
from scipy.linalg import block_diag
from src.data.misc import pair_attention_transform, transform_dataset_to_pair_dataset_torch

class CustomDataLoader(DataLoader):
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
        # batch is a list of pairs and targets
        idx_a, idx_b, target = zip(*[(x[0], x[1], x[2]) for x in batch])
        if self.mode == 'default':
            data = self.get_data_from_indices(idx_a, idx_b, unique=True)
            idx_a, idx_b = self.reindex_ids(idx_a, idx_b)

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
            data_batch = [data_a, data_b]
        elif self.mode == 'nagsl_attention':
            data_a = self.get_data_from_indices(idx_a, [], unique=False)
            data_b = self.get_data_from_indices(idx_b, [], unique=False)

            data_a = Batch.from_data_list(data_a)
            data_b = Batch.from_data_list(data_b)
            data_batch = pair_attention_transform((data_a, data_b), torch.tensor(target), self.config)
        elif self.mode == 'my_attention':
            batch_with_connected_graphs = transform_dataset_to_pair_dataset_torch(self.entire_dataset, batch, self.config)
            num_nodes = [g.num_nodes for g in batch_with_connected_graphs]
            adjacency_matrices = [to_dense_adj(g.edge_index).squeeze(0) for g in batch_with_connected_graphs]
            attention_data = block_diag(*adjacency_matrices)
            document_id = torch.repeat_interleave(torch.arange(len(num_nodes)), torch.tensor(num_nodes), dim=0, output_size=sum(num_nodes))
            data = self.get_data_from_indices(idx_a, idx_b, unique=True)
            idx_a, idx_b = self.reindex_ids(idx_a, idx_b)

            # Create a DataBatch object
            data_batch = Batch.from_data_list(data)

            # Add idx_a and idx_b to the DataBatch object
            data_batch.idx_a = torch.tensor(idx_a)
            data_batch.idx_b = torch.tensor(idx_b)
            data_batch.y = torch.tensor(target)
            data_batch.document_id = document_id
            data_batch.attention_data = torch.tensor(attention_data)
            data_batch.unique = self.batch_size
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return data_batch

    def get_data_from_indices(self, idx_a, idx_b, unique=False):
        if unique:
            ids = np.unique(np.concatenate((idx_a, idx_b)))
        else:
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