import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, ReLU
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential, Tanh, PReLU
from torch_geometric.nn import GCNConv, AGNNConv, GINConv, global_add_pool, global_mean_pool, EdgeConv, GraphConv, global_add_pool

class RankGNN(torch.nn.Module):
    ''' Pairwise GraphConvolutionNetwork
        data are graphs
        pairs are referenced by the idx_*
    '''
    def __init__(self, num_node_features=9, device='cpu', config=None):
        super().__init__()
        self.num_node_features = num_node_features
        self.device = device
        print(self.device)
        self.config = config
        self.convIn = GCNConv(self.num_node_features, config['model_units'])
        self.conv2 = GCNConv(config['model_units'], config['model_units'])
        self.convOut = GCNConv(config['model_units'], config['model_units'])
        self.fc1 = Linear(config['model_units'], config['model_units'])
        self.fc2 = Linear(config['model_units'], 32)
        self.fc3 = Linear(32, 1)  # Output 1 for regression
        self.dropout = Dropout(config['model_dropout'])

    def pref_lookup(self, util, idx_a, idx_b):
        util = util.squeeze()
        # print(util.size())
        idx_a = idx_a.to(torch.int64)
        # print(idx_a.size())
        idx_b = idx_b.to(torch.int64)
        pref_a = torch.gather(util, 0, idx_a)
        pref_b = torch.gather(util, 0, idx_b)
        return pref_a, pref_b

    def forward(self, data):
        x, edge_index, batch, idx_a, idx_b = data.x, data.edge_index, data.batch, data.idx_a, data.idx_b
        x = x.type(torch.FloatTensor).to(self.device)
        x = self.convIn(x, edge_index)
        x = F.tanh(x)
        x = self.conv2(x, edge_index)
        x = F.tanh(x)
        x = self.dropout(x)
        x = self.convOut(x, edge_index)
        x = F.tanh(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.tanh(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = F.tanh(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)

        x_util = global_add_pool(x, batch)

        x_a, x_b = self.pref_lookup(x_util, idx_a, idx_b)
        out = x_b - x_a

        return out, x_util

class PairRankGNN(torch.nn.Module):
    ''' Pairwise GraphConvolution Network
        data represents the fully connected graph pairs (no idx_* needed)
    '''
    def __init__(self, num_node_features=9, device='cpu', config=None):
        super(PairRankGNN, self).__init__()
        self.num_node_features = num_node_features
        self.device = device
        self.config = config
        self.conv1 = GCNConv(self.num_node_features, config['model_units'])
        self.conv2 = GCNConv(config['model_units'], config['model_units'])
        self.fc = Linear(config['model_units'], 1)  # Output 1 for regression

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.type(torch.FloatTensor).to(self.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        out = global_mean_pool(x, batch)
        return out

class PairRankGNN2(torch.nn.Module):
    ''' Pairwise GraphConvolution Network
        data represents the fully connected graph pairs (no idx_* needed)
        new edge connections are stored separately
    '''
    def __init__(self, num_node_features=9, device='cpu', config=None):
        super(PairRankGNN2, self).__init__()
        self.num_node_features = num_node_features
        self.device = device
        self.config = config
        self.conv1 = GCNConv(self.num_node_features, config['model_units'])
        self.edge1 = EdgeConv(config['model_units'], config['model_units'])
        self.conv2 = GCNConv(config['model_units'], 32)
        self.edge2 = EdgeConv(32, 32)
        self.fc = Linear(32, 1)  # Output 1 for regression

    def forward(self, data):
        x, edge_index, adj, batch= data.x, data.edge_index, data.adj, data.batch
        x = x.type(torch.FloatTensor).to(self.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.edge1(x, adj)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.edge2(x, adj)
        x = F.relu(x)
        x = self.fc(x)
        out = global_mean_pool(x, batch)
        return out