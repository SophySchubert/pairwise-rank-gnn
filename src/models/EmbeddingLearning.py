import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.dense import DenseGCNConv, DenseGINConv, DenseSAGEConv


class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
        self.config = config

        hidden_size = self.config['embedding_size']
        ffn_size = self.config['encoder_ffn_size']
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config, q=None, k=None):
        super(MultiHeadAttention, self).__init__()
        self.config = config

        self.num_heads = num_heads = self.config['n_heads']
        embedding_size = self.config['embedding_size']

        self.att_size = embedding_size
        self.scale = embedding_size ** -0.5

        self.linear_q = q if q else nn.Linear(embedding_size, num_heads * embedding_size, bias=self.config['msa_bias'])
        self.linear_k = k if k else nn.Linear(embedding_size, num_heads * embedding_size, bias=self.config['msa_bias'])
        self.linear_v = nn.Linear(embedding_size, num_heads * embedding_size, bias=self.config['msa_bias'])
        self.att_dropout = nn.Dropout(self.config['dropout'])

        self.output_layer = nn.Linear(num_heads * embedding_size, embedding_size)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        torch.nn.init.xavier_uniform_(self.linear_v.weight)

    def forward(self, x, dist=None, mask=None):
        d_k = self.att_size
        d_v = self.att_size
        batch_size = x.size(0)

        if self.config['encoder_mask']:
            q = self.linear_q(x).view(batch_size, -1, self.num_heads, d_k).transpose(-2, -3)
            k = self.linear_k(x).view(batch_size, -1, self.num_heads, d_k).transpose(-2, -3).transpose(-1, -2)
            v = self.linear_v(x).view(batch_size, -1, self.num_heads, d_v).transpose(-2, -3)
            q = torch.einsum('bhne,bn->bhne', q, mask)
            k = torch.einsum('bhen,bn->bhen', k, mask)
            v = torch.einsum('bhne,bn->bhne', v, mask)
        else:
            q = self.linear_q(x).view(batch_size, self.num_heads, -1, d_k)
            k = self.linear_k(x).view(batch_size, self.num_heads, -1, d_k).transpose(-1, -2)
            v = self.linear_v(x).view(batch_size, self.num_heads, -1, d_v)

        q = q * self.scale
        a = torch.matmul(q, k)


        dist_decay = 1
        if dist is not None:
            dist *= dist_decay
            a += torch.stack([dist] * self.config['n_heads'], dim=1).to(self.config['device'])

        # masked softmax
        if self.config['encoder_mask']:
            attention_mask = torch.einsum('ij,ik->ijk', mask, mask)
            a = a.transpose(0, 1).masked_fill(attention_mask == 0, -1e9)
            a = torch.softmax(a, dim=3).masked_fill(attention_mask == 0, 0).transpose(0, 1)
        else:
            a = torch.softmax(a, dim=3)

        a = self.att_dropout(a)

        y = a.matmul(v).transpose(-2, -3).contiguous().view(batch_size, -1, self.num_heads * d_v)
        y = self.output_layer(y)

        if self.config['encoder_mask']:
            y = torch.einsum('bne,bn->bne', y, mask)

        return y


class GCNTransformerEncoder(nn.Module):
    def __init__(self, config, q=None, k=None):
        super(GCNTransformerEncoder, self).__init__()
        self.config = config

        if self.config['GNN'] == 'GCN':
            self.GCN_first = DenseGCNConv(self.config['num_node_features'], self.config['embedding_size'])
            self.GCN_second = DenseGCNConv(self.config['embedding_size'], self.config['embedding_size'])
            self.GCN_third = DenseGCNConv(self.config['embedding_size'], self.config['embedding_size'])
        elif self.config['GNN'] == 'SAGE':
            self.GCN_first = DenseSAGEConv(self.config['num_node_features'], self.config['embedding_size'])
            self.GCN_second = DenseSAGEConv(self.config['embedding_size'], self.config['embedding_size'])
            self.GCN_third = DenseSAGEConv(self.config['embedding_size'], self.config['embedding_size'])
        elif self.config['GNN'] == 'GIN':
            self.GCN_first = DenseGINConv(nn.Linear(self.config['num_node_features'], self.config['embedding_size']))
            self.GCN_second = DenseGINConv(nn.Linear(self.config['embedding_size'], self.config['embedding_size']))
            self.GCN_third = DenseGINConv(nn.Linear(self.config['embedding_size'], self.config['embedding_size']))
        else:
            raise ValueError('GNN argument error')

        self.d_k = self.config['embedding_size']

        # torch.nn.init.xavier_uniform_(self.GCN_first.lin.weight)
        # torch.nn.init.xavier_uniform_(self.GCN_second.lin.weight)
        # torch.nn.init.xavier_uniform_(self.GCN_third.lin.weight)

        self.self_attention_norm = nn.LayerNorm(self.config['embedding_size'])
        self.self_attention = MultiHeadAttention(self.config, q, k)
        self.self_attention_dropout = nn.Dropout(self.config['dropout'])

        self.ffn_norm = nn.LayerNorm(self.config['embedding_size'])
        self.ffn = FeedForwardNetwork(self.config)
        self.ffn_dropout = nn.Dropout(self.config['dropout'])

    def forward(self, x, adj, mask, dist=None):
        first_gcn_result = F.relu(self.GCN_first(x, adj, mask))
        second_gcn_result = F.relu(self.GCN_second(first_gcn_result, adj, mask))
        gcn_result = F.relu(self.GCN_third(second_gcn_result, adj, mask))

        if self.config['GT_res']:
            gcn_result = gcn_result + first_gcn_result + second_gcn_result

        self_att_result = self.self_attention_norm(gcn_result)
        self_att_result = self.self_attention(self_att_result, dist, mask)
        self_att_result = self.self_attention_dropout(self_att_result)
        self_att_result = gcn_result + self_att_result

        ffn_result = self.ffn_norm(self_att_result)
        ffn_result = self.ffn(ffn_result)
        if self.config['encoder_mask']:
            ffn_result = torch.einsum('bne,bn->bne', ffn_result, mask)
        ffn_result = self.ffn_dropout(ffn_result)
        self_att_result = self_att_result + ffn_result

        if self.config['GT_res']:
            encoder_result = gcn_result + self_att_result
        else:
            encoder_result = self_att_result

        return encoder_result
