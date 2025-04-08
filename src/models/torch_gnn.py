import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, EdgeConv

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
        for i in range(self.config['model_layers'] - 2):
            x = self.conv2(x, edge_index)
            x = F.tanh(x)
            x = self.dropout(x)
        x = self.convOut(x, edge_index)
        x = F.tanh(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        x = self.dropout(x)

        x_util = global_mean_pool(x, batch)

        x_a, x_b = self.pref_lookup(x_util, idx_a, idx_b)
        out = x_b - x_a
        out = F.sigmoid(out)

        return out, x_util

class RankGAN(torch.nn.Module):
    ''' Pairwise GraphAttentionNetwork
        data are graphs
        pairs are given by the batch and then compared via pref_lookup()
    '''
    def __init__(self, num_node_features=9, device='cpu', config=None):
        super().__init__()
        self.num_node_features = num_node_features
        self.device = device
        self.config = config
        self.convIn = GATConv(self.num_node_features, config['model_units'])
        self.conv2 = GATConv(config['model_units'], config['model_units'])
        self.convOut = GATConv(config['model_units'], config['model_units'])
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
        data_a, data_b = data[0], data[1]
        x_a, edge_index_a, batch_a = data_a.x, data_a.edge_index, data_a.batch
        x_b, edge_index_b, batch_b = data_b.x, data_b.edge_index, data_b.batch
        x_a = x_a.type(torch.FloatTensor).to(self.device)
        x_b = x_b.type(torch.FloatTensor).to(self.device)
        # Network a
        x_a = self.convIn(x_a, edge_index_a)
        x_a = F.tanh(x_a)
        for i in range(self.config['model_layers'] - 2):
            x_a = self.conv2(x_a, edge_index_a)
            x_a = F.tanh(x_a)
            x_a = self.dropout(x_a)
        x_a = self.convOut(x_a, edge_index_a)
        x_a = F.tanh(x_a)
        x_a = self.dropout(x_a)
        x_a = self.fc1(x_a)
        x_a = F.tanh(x_a)
        x_a = self.fc2(x_a)
        x_a = F.tanh(x_a)
        x_a = self.fc3(x_a)
        x_a = self.dropout(x_a)
        # Network b
        x_b = self.convIn(x_b, edge_index_b)
        x_b = F.tanh(x_b)
        for i in range(self.config['model_layers'] - 2):
            x_b = self.conv2(x_b, edge_index_b)
            x_b = F.tanh(x_b)
            x_b = self.dropout(x_b)
        x_b = self.convOut(x_b, edge_index_b)
        x_b = F.tanh(x_b)
        x_b = self.dropout(x_b)
        x_b = self.fc1(x_b)
        x_b = F.tanh(x_b)
        x_b = self.fc2(x_b)
        x_b = F.tanh(x_b)
        x_b = self.fc3(x_b)
        x_b = self.dropout(x_b)

        x_a_util = global_mean_pool(x_a, batch_a)
        x_b_util = global_mean_pool(x_b, batch_b)


        out = x_b_util - x_a_util

        return out

class PairRankGNN(torch.nn.Module):
    ''' Pairwise GraphConvolution Network
        data represents the fully connected graph pairs (no idx_* needed)
    '''
    def __init__(self, num_node_features=9, device='cpu', config=None):
        raise("IMPLEMENT ME ")
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
        raise("IMPLEMENT ME ")
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