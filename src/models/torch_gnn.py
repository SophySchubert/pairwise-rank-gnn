import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, Conv2d
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, _mask_mod_signature, noop_mask
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, EdgeConv

from models.misc import create_document_mask

class RankGNN(torch.nn.Module):
    ''' Pairwise GraphConvolutionNetwork
        data are graphs
        pairs are referenced by the idx_*
    '''
    def __init__(self, num_node_features=9, device='cpu', config=None):
        super().__init__()
        self.num_node_features = num_node_features
        self.device = device
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
        idx_a = idx_a.to(torch.int64)
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
        idx_a = idx_a.to(torch.int64)
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
        out = F.sigmoid(out)

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
        out = F.sigmoid(out)
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
        out = F.sigmoid(out)
        return out

class RANet(torch.nn.Module):
    ''' Rank Attention Network
    '''
    def __init__(self, config=None):
        super(RANet, self).__init__()
        self.device = config['device']
        self.config = config
        self.convIn = GCNConv(config['num_node_features'], config['model_units'])
        self.conv2 = GCNConv(config['model_units'], config['model_units'])
        self.convOut = GCNConv(config['model_units'], config['model_units'])
        self.fc1 = Linear(config['model_units'], config['model_units'])
        self.fc2 = Linear(config['model_units'], 32)
        self.fc3 = Linear(32, 1)  # Output 1 for regression
        self.dropout = Dropout(config['model_dropout'])

        self.query_mlp = Sequential(
            Conv2d(32, 64, kernel_size=(1, 1)),
            torch.nn.ReLU(),
            Conv2d(64, 32, kernel_size=(1, 1)),
            torch.nn.ReLU(),
            Conv2d(32, 32, kernel_size=(1, 1)),
            torch.nn.ReLU()
        )
        self.key_mlp = Sequential(
            Conv2d(32, 64, kernel_size=(1, 1)),
            torch.nn.ReLU(),
            Conv2d(64, 32, kernel_size=(1, 1)),
            torch.nn.ReLU(),
            Conv2d(32, 32, kernel_size=(1, 1)),
            torch.nn.ReLU()
        )
        self.value_mlp = Sequential(
            Conv2d(32, 64, kernel_size=(1, 1)),
            torch.nn.ReLU(),
            Conv2d(64, 32, kernel_size=(1, 1)),
            torch.nn.ReLU(),
            Conv2d(32, 32, kernel_size=(1, 1)),
            torch.nn.ReLU()
        )

    def forward(self, data):
        # Compute block mask at beginning of forwards due to changing every batch
        x, edge_index, batch, document_id, unique = data.x, data.edge_index, data.batch, data.document_id, data.unique
        causal_mask = self.generate_doc_mask_mod(document_id, unique)
        ranking_mask = create_block_mask(mask_mod=causal_mask, B=1, H=1, Q_LEN=x.shape[0], KV_LEN=x.shape[0], device=self.device)
        x = x.type(torch.FloatTensor).to(self.device)

        x = self.convIn(x, edge_index)
        x = F.tanh(x)
        x = self.attention_layer(x, ranking_mask)
        for i in range(self.config['model_layers'] - 2):
            x = self.conv2(x, edge_index)
            x = F.tanh(x)
            x = self.dropout(x)
            x = self.attention_layer(x, ranking_mask)
        x = self.convOut(x, edge_index)
        x = F.tanh(x)
        x = self.dropout(x)
        x = self.attention_layer(x, ranking_mask)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        x = self.dropout(x)

        out = global_mean_pool(x, batch)
        out = F.sigmoid(out)

        return out

    def attention_layer(self, x, mask):
        a_q = self.query_mlp(x.view(x.shape[0], x.shape[1], 1, 1))
        a_k = self.key_mlp(x.view(x.shape[0], x.shape[1], 1, 1))
        a_v = self.value_mlp(x.view(x.shape[0], x.shape[1], 1, 1))
        x = flex_attention(query=a_q.permute(2,3,0,1),
                           key=a_k.permute(2,3,0,1),
                           value=a_v.permute(2,3,0,1),
                           block_mask=mask)
        return x.squeeze(0).squeeze(0)

    def pref_lookup(self, util, idx_a, idx_b):
        util = util.squeeze()
        idx_a = idx_a.to(torch.int64)
        idx_b = idx_b.to(torch.int64)

        pref_a = torch.gather(util, 0, idx_a)
        pref_b = torch.gather(util, 0, idx_b)
        return pref_a, pref_b

    def generate_doc_mask_mod(self, document_id: torch.Tensor, unique: int) -> _mask_mod_signature:
        """Generates mask mods that apply to inputs to flex attention in the sequence stacked
        format.

        Args:
            docment_id: A tensor that contains a unique id for each graph and has the length of the number of nodes in the graph-batch. Each id has the
                        length of the number of nodes in the graph.

        Note:
            What is the sequence stacked format? When assembling batches of inputs, we
            take multiple sequences and stack them together to form 1 large sequence. We then
            use masking to ensure that the attention scores are only applied to tokens within
            the different graphs but of the same pair.
        """
        def doc_mask_mod(b, h, q_idx, kv_idx):
            dif_doc = (document_id[q_idx] != document_id[kv_idx])
            operation = False
            for i in range(0, unique, 2):
                operation = operation | (((document_id[q_idx] == i) & (document_id[kv_idx] == i+1)) | (
                            (document_id[q_idx] == i+1) & (document_id[kv_idx] == i)))
            inner_mask = noop_mask(b, h, q_idx, kv_idx)

            return dif_doc & operation & inner_mask

        return doc_mask_mod