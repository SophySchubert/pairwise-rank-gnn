import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ModuleList
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, _mask_mod_signature, noop_mask
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, EdgeConv
import numpy as np
'''
Graph Neural Networks
'''

class RankGNN(torch.nn.Module):
    ''' Pairwise GraphConvolutionNetwork
        data are graphs
        pairs are referenced by the idx_*
        mode in config: 'default'
    '''
    def __init__(self, num_node_features=9, device='cpu', config=None):
        super().__init__()
        self.num_node_features = num_node_features
        self.device = device
        self.config = config
        self.convIn = GCNConv(self.num_node_features, config['model_units'])
        self.convs = ModuleList([
            GCNConv(config['model_units'], config['model_units']).to(self.device)
            for _ in range(config['model_layers'] - 1)
        ])
        self.fc1 = Linear(config['model_units'], config['model_units'])
        self.fc2 = Linear(config['model_units'], 32)
        self.fc3 = Linear(32, 1)  # Output 1 for regression

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

        # Convolutional layers for graph processing
        x = self.convIn(x, edge_index)
        x = F.tanh(x)
        for i in range(self.config['model_layers'] - 1):
            x = self.convs[i](x, edge_index)
            x = F.tanh(x)
            x = F.dropout(x, p=self.config['model_dropout'], training=self.training)
        # Fully connected MLP for further extraction of features
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        x = F.dropout(x, p=self.config['model_dropout'], training=self.training)

        # Global pooling to get a single vector for each graph
        x_util = global_mean_pool(x, batch)

        # Actual pairwise comparing module of the NN
        x_a, x_b = self.pref_lookup(x_util, idx_a, idx_b)# Lookup of preferences for the pairs
        out = x_b - x_a # Subtraction of the two preferences
        out = F.sigmoid(out) # Last activation to squash the output to [0,1]

        return out, x_util

class RankGAT(torch.nn.Module):
    ''' Pairwise GraphAttentionNetwork
        data are graphs
        pairs are given by the batch and then compared via subtraction
        mode in config: 'gat_attention'
    '''
    def __init__(self, num_node_features=9, device='cpu', config=None):
        super().__init__()
        self.num_node_features = num_node_features
        self.device = device
        self.config = config
        self.convIn = GATConv(self.num_node_features, config['model_units'])
        self.convs = ModuleList([
            GCNConv(config['model_units'], config['model_units']).to(self.device)
            for _ in range(config['model_layers'] - 1)
        ])
        self.fc1 = Linear(config['model_units'], config['model_units'])
        self.fc2 = Linear(config['model_units'], 32)
        self.fc3 = Linear(32, 1)  # Output 1 for regression

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
        for i in range(self.config['model_layers'] - 1):
            x_a = self.convs[i](x_b, edge_index_a)
            x_a = F.tanh(x_a)
            x_a = F.dropout(x_a, p=self.config['model_dropout'], training=self.training)
        x_a = self.fc1(x_a)
        x_a = F.tanh(x_a)
        x_a = self.fc2(x_a)
        x_a = F.tanh(x_a)
        x_a = self.fc3(x_a)
        x_a = F.dropout(x_a, p=self.config['model_dropout'], training=self.training)
        # Network b
        x_b = self.convIn(x_b, edge_index_b)
        x_b = F.tanh(x_b)
        for i in range(self.config['model_layers'] - 1):
            x_b = self.convs[i](x_b, edge_index_b)
            x_b = F.tanh(x_b)
            x_b = F.dropout(x_b, p=self.config['model_dropout'], training=self.training)
        x_b = self.fc1(x_b)
        x_b = F.tanh(x_b)
        x_b = self.fc2(x_b)
        x_b = F.tanh(x_b)
        x_b = self.fc3(x_b)
        x_b = F.dropout(x_b, p=self.config['model_dropout'], training=self.training)

        x_a_util = global_mean_pool(x_a, batch_a)
        x_b_util = global_mean_pool(x_b, batch_b)


        out = x_b_util - x_a_util
        out = F.sigmoid(out)

        return out

class PairRankGNN(torch.nn.Module):
    ''' Pairwise GraphConvolution Network
        data represents the fully connected graph pairs (no idx_* needed)
        mode in config: 'fc'
    '''
    def __init__(self, num_node_features=9, device='cpu', config=None):
        super(PairRankGNN, self).__init__()
        self.num_node_features = num_node_features
        self.device = device
        self.config = config
        self.convIn = GCNConv(self.num_node_features, config['model_units'])
        self.convs = ModuleList([
            GCNConv(config['model_units'], config['model_units']).to(self.device)
            for _ in range(config['model_layers'] - 1)
        ])
        self.fc1 = Linear(config['model_units'], config['model_units'])
        self.fc2 = Linear(config['model_units'], 32)
        self.fc3 = Linear(32, 1)  # Output 1 for regression

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.type(torch.FloatTensor).to(self.device)
        x = self.convIn(x, edge_index)
        x = F.tanh(x)
        for i in range(self.config['model_layers'] - 1):
            x = self.convs[i](x, edge_index)
            x = F.tanh(x)
            x = F.dropout(x, p=self.config['model_dropout'], training=self.training)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        x = F.dropout(x, p=self.config['model_dropout'], training=self.training)

        x = global_mean_pool(x, batch)
        out = F.sigmoid(x)

        return out

class PairRankGNN2(torch.nn.Module):
    ''' Pairwise GraphConvolution Network
        data represents the fully connected graph pairs (no idx_* needed)
        Also uses EdgeConv-Layers
        mode in config: 'fc_extra'
    '''
    def __init__(self, num_node_features=9, device='cpu', config=None):
        super(PairRankGNN2, self).__init__()
        self.num_node_features = num_node_features
        self.device = device
        self.config = config
        self.conv1 = GCNConv(self.num_node_features, config['model_units'])
        self.convs = ModuleList([
            GCNConv(config['model_units'], config['model_units']).to(self.device)
            for _ in range(config['model_layers'] - 1)
        ])
        self.conv2 = GCNConv(config['model_units'], config['model_units'])
        self.edges = ModuleList([
            EdgeConv(nn=Sequential(Linear(config['model_units'] * 2, config['model_units']),
                                   ReLU(), Linear(config['model_units'], config['model_units'])))
            for _ in range(config['model_layers'])
        ])
        self.fc1 = Linear(config['model_units'], config['model_units'])
        self.fc2 = Linear(config['model_units'], 32)
        self.fc3 = Linear(32, 1)  # Output 1 for regression

    def forward(self, data):
        x, edge_index, batch= data.x, data.edge_index, data.batch
        x = x.type(torch.FloatTensor).to(self.device)

        x = x.type(torch.FloatTensor).to(self.device)
        x = self.conv1(x, edge_index)
        x = F.tanh(x)
        x = self.edges[0](x, edge_index)
        x = F.tanh(x)
        for i in range(self.config['model_layers'] - 1):
            x = self.convs[i](x, edge_index)
            x = F.tanh(x)
            x = F.dropout(x, p=self.config['model_dropout'], training=self.training)
            x = self.edges[1+i](x, edge_index)
            x = F.tanh(x)
            x = F.dropout(x, p=self.config['model_dropout'], training=self.training)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        x = F.dropout(x, p=self.config['model_dropout'], training=self.training)

        x = global_mean_pool(x, batch)
        out = F.sigmoid(x)

        return out

class RANet(torch.nn.Module):
    '''
    Ranking Mask Attention Network - main part of this research and framework
    data are graphs
    pairs are referenced by the idx_*
    utilises flex_attention from pytorch (version>=2.5)
    mode in config: 'rank_mask'
    '''
    def __init__(self, num_node_features=9, device='cpu', config=None):
        super().__init__()
        self.num_node_features = num_node_features
        self.device = device
        self.config = config
        self.convIn = GCNConv(self.num_node_features, config['model_units'])
        self.convs = ModuleList([
            GCNConv(config['model_units'], config['model_units']).to(self.device)
            for _ in range(config['model_layers'] - 1)
        ])
        self.fc1 = Linear(config['model_units'], config['model_units'])
        self.fc2 = Linear(config['model_units'], 32)
        self.fc3 = Linear(32, 1)  # Output 1 for regression
        # QKV MLPs for attention
        self.qkv_mlps = ModuleList([
            ModuleList([
                Sequential(
                    Linear(config['model_units'], config['model_units2']), torch.nn.ReLU(),
                    Linear(config['model_units2'], config['model_units2']), torch.nn.ReLU(),
                    Linear(config['model_units2'], config['model_units']), torch.nn.ReLU()
                ),
                Sequential(
                    Linear(config['model_units'], config['model_units2']), torch.nn.ReLU(),
                    Linear(config['model_units2'], config['model_units2']), torch.nn.ReLU(),
                    Linear(config['model_units2'], config['model_units']), torch.nn.ReLU()
                ),
                Sequential(
                    Linear(config['model_units'], config['model_units2']), torch.nn.ReLU(),
                    Linear(config['model_units2'], config['model_units2']), torch.nn.ReLU(),
                    Linear(config['model_units2'], config['model_units']), torch.nn.ReLU()
                )
            ])
            for _ in range(config['model_layers'])
        ])

    def create_documen_id(self, num_nodes_g_1, num_nodes_g_2):
        ids = np.empty((len(num_nodes_g_1) + len(num_nodes_g_2)))
        ids[0::2] = num_nodes_g_1.cpu().numpy()
        ids[1::2] = num_nodes_g_2.cpu().numpy()
        document_id = torch.repeat_interleave(torch.arange(len(ids)),
                                              torch.tensor(ids, dtype=torch.long),
                                              dim=0,
                                              output_size=int(sum(ids)))
        return document_id.to(self.device)

    def forward(self, data):
        x, edge_index, batch, num_nodes_g_1, num_nodes_g_2 = data.x, data.edge_index, data.batch, data.g_1, data.g_2
        document_id = self.create_documen_id(num_nodes_g_1, num_nodes_g_2)
        causal_mask = self.generate_doc_mask_mod(document_id, len(num_nodes_g_1))
        ranking_mask = create_block_mask(mask_mod=causal_mask, B=1, H=1, Q_LEN=x.shape[0], KV_LEN=x.shape[0], device=self.device)
        x = x.type(torch.FloatTensor).to(self.device)

        x = self.convIn(x, edge_index)
        x = F.tanh(x)
        # Attention block start
        q_mlp, k_mlp, v_mlp = self.qkv_mlps[0][0], self.qkv_mlps[0][1], self.qkv_mlps[0][2]
        a_q = q_mlp(x)#torch.ones(x.shape[0], x.shape[1], device=self.device)#q_mlp(x)
        a_k = k_mlp(x)#torch.ones(x.shape[0], x.shape[1], device=self.device)#k_mlp(x)
        a_v = v_mlp(x)
        x = flex_attention(query=a_q.unsqueeze(0).unsqueeze(0),
                           key=a_k.unsqueeze(0).unsqueeze(0),
                           value=a_v.unsqueeze(0).unsqueeze(0),
                           block_mask=ranking_mask).squeeze(0).squeeze(0)
        # Attention block end
        for i in range(self.config['model_layers'] - 1):
            x = self.convs[i](x, edge_index)
            x = F.tanh(x)
            x = F.dropout(x, p=self.config['model_dropout'], training=self.training)
            # q_mlp, k_mlp, v_mlp = self.qkv_mlps[1+i][0], self.qkv_mlps[1+i][1], self.qkv_mlps[1+i][2]
            # a_q = q_mlp(x)
            # a_k = k_mlp(x)
            # a_v = v_mlp(x)
            # x = flex_attention(query=a_q.unsqueeze(0).unsqueeze(0),
            #                    key=a_k.unsqueeze(0).unsqueeze(0),
            #                    value=a_v.unsqueeze(0).unsqueeze(0),
            #                    block_mask=ranking_mask).squeeze(0).squeeze(0)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        x = F.dropout(x, p=self.config['model_dropout'], training=self.training)

        x = global_mean_pool(x, batch)
        out = F.sigmoid(x)

        return out

    def generate_doc_mask_mod(self, document_id: torch.Tensor, unique: int) -> _mask_mod_signature:
        """Generates mask mods that apply to inputs to flex attention in the sequence stacked
        format.

        Args:
            docment_id: A tensor that contains a unique id for each graph and has the length of the number of nodes in the graph-batch. Each id has the
                        length of the number of nodes in the graph.
            unique: The number of unique document ids in the batch. This is the number of graphs in the batch and original batch size.

        Note:
            What is the sequence stacked format? When assembling batches of inputs, we
            take multiple sequences and stack them together to form 1 large sequence. We then
            use masking to ensure that the attention scores are only applied to tokens within
            the different graphs but of the same pair.
        """
        def doc_mask_mod(b, h, q_idx, kv_idx):
            if self.config['attention'] == 'cross':
            # calculates a mask mod that only is True in areas where the connection between graphs is.
            # The rest is False
                diff_graph = (document_id[q_idx] != document_id[kv_idx]) # the first pair
                operation = False
                for i in range(0, unique, 2):
                    # all following pair id combinations are calculated in this loop
                    operation = operation | (((document_id[q_idx] == i) & (document_id[kv_idx] == i+1)) | (
                                (document_id[q_idx] == i+1) & (document_id[kv_idx] == i)))
                inner_mask = noop_mask(b, h, q_idx, kv_idx) # simple noop_mask
                return diff_graph & operation & inner_mask # combine all 3 masks
            elif self.config['attention'] == 'self':
                same_graph = (document_id[q_idx] == document_id[kv_idx])
                return same_graph
            else:
                raise ValueError(f"Unknown attention type: {self.config['attention']}")

        return doc_mask_mod