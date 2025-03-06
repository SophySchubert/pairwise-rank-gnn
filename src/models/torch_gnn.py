import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class RGNN(torch.nn.Module):
    ''' Pairwise GraphConvolutionNetwork
        data are graphs
        pairs are referenced by the idx_*
    '''
    def __init__(self, num_node_features=9, device='cpu'):
        super(RGNN, self).__init__()
        self.num_node_features = num_node_features
        self.device = device
        self.conv1 = GCNConv(self.num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = torch.nn.Linear(64, 1)  # Output 1 for regression

    def pref_lookup(self, util, idx_a, idx_b):
        util = util.squeeze()
        pref_a = torch.gather(util, 0, idx_a)
        pref_b = torch.gather(util, 0, idx_b)
        return pref_a, pref_b

    def forward(self, data):
        x, edge_index, batch, idx_a, idx_b = data.x, data.edge_index, data.batch, data.idx_a, data.idx_b
        x = x.type(torch.FloatTensor).to(self.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.fc(x)
        x_util = global_mean_pool(x, batch)

        x_a, x_b = self.pref_lookup(x_util, idx_a, idx_b)
        out = x_b - x_a

        return out, x_util

class PRGNN(torch.nn.Module):
    ''' Pairwise GraphConvolution Network
        data represents the fully connected graph pairs (no idx_* needed)
    '''
    def __init__(self, num_node_features=9, device='cpu'):
        super(PRGNN, self).__init__()
        self.num_node_features = num_node_features
        self.device = device
        self.conv1 = GCNConv(self.num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = torch.nn.Linear(64, 1)  # Output 1 for regression

    def forward(self, data):
        x, edge_index, batch, idx_a, idx_b = data.x, data.edge_index, data.batch
        x = x.type(torch.FloatTensor).to(self.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.fc(x)
        out = global_mean_pool(x, batch)

        return out