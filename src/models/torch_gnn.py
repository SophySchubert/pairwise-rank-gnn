import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, ReLU
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential, Tanh
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, EdgeConv, GraphConv, global_add_pool

class RankGNN(torch.nn.Module):
    ''' Pairwise GraphConvolutionNetwork
        data are graphs
        pairs are referenced by the idx_*
    '''
    def __init__(self, num_node_features=9, device='cpu', config=None):
        super(RankGNN, self).__init__()
        self.num_node_features = num_node_features
        self.device = device
        self.config = config
        self.conv1 = GCNConv(self.num_node_features, config['model_units'])
        self.conv2 = GCNConv(config['model_units'], 32)
        self.fc1 = Linear(32, 32)
        self.fc2 = Linear(32, 1)  # Output 1 for regression
        self.dropout = Dropout(config['model_dropout'])
        # self.convs = torch.nn.ModuleList()
        # self.batch_norms = torch.nn.ModuleList()
        # in_channels = self.num_node_features
        # for i in range(2):
        #     mlp = Sequential(
        #         Linear(in_channels, 2 * config['model_units']),
        #         BatchNorm(2 * config['model_units']),
        #         Tanh(),
        #         Linear(2 * config['model_units'], config['model_units']),
        #     )
        #     conv = GINConv(mlp, train_eps=True)
        #
        #     self.convs.append(conv)
        #     self.batch_norms.append(BatchNorm(config['model_units']))
        #
        #     in_channels = config['model_units']
        #
        # self.lin1 = Linear(config['model_units'], config['model_units'])
        # self.batch_norm1 = BatchNorm(config['model_units'])
        # self.lin2 = Linear(config['model_units'], 1)

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
        # print("network")
        x, edge_index, batch, idx_a, idx_b = data.x, data.edge_index, data.batch, data.idx_a, data.idx_b
        x = x.type(torch.FloatTensor).to(self.device)
        x = self.conv1(x, edge_index)
        x = Tanh()(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = Tanh()(x)
        x = self.dropout(x)
        x = self.fc1(x)
        # x = ReLU()(x)
        x = self.fc2(x)
        # for conv, batch_norm in zip(self.convs, self.batch_norms):
        #     x = F.relu(batch_norm(conv(x, edge_index)))
        # x = global_mean_pool(x, batch)
        # x = F.relu(self.batch_norm1(self.lin1(x)))
        # x = F.dropout(x, p=self.config['model_dropout'], training=self.training)
        # x_util = self.lin2(x)
        x_util = global_mean_pool(x, batch)

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