import torch
import torch.nn as nn


class CrossTransformer(nn.Module):
    def __init__(self, config, q=None, k=None):
        super(CrossTransformer, self).__init__()
        self.config = config

        self.cross_attention = CrossAttention(self.config, q, k).to(self.config['device'])


    def forward(self, embeddings_i, mask_i, embeddings_j, mask_j, mask_ij=None):

        y = self.cross_attention(embeddings_i, mask_i, embeddings_j, mask_j, mask_ij)

        return y


class CrossAttention(nn.Module):
    def __init__(self, config, q=None, k=None):
        super(CrossAttention, self).__init__()
        self.config = config
        self.scale = self.config['embedding_size'] ** -0.5

        self.linear_q = q if q else nn.Linear(self.config['embedding_size'], self.config['n_heads'] * self.config['embedding_size'], bias=self.config['msa_bias'])
        self.linear_k = k if k else nn.Linear(self.config['embedding_size'], self.config['n_heads'] * self.config['embedding_size'], bias=self.config['msa_bias'])

    def forward(self, embeddings_i, mask_i, embeddings_j, mask_j, mask_ij=None):
        batch_size = embeddings_i.size(0)

        q_i = self.linear_q(embeddings_i).view(batch_size, -1, self.config['n_heads'], self.config['embedding_size']).transpose(-2, -3)
        k_i = self.linear_k(embeddings_i).view(batch_size, -1, self.config['n_heads'], self.config['embedding_size']).transpose(-2, -3).transpose(-1, -2)
        q_j = self.linear_q(embeddings_j).view(batch_size, -1, self.config['n_heads'], self.config['embedding_size']).transpose(-2, -3)
        k_j = self.linear_k(embeddings_j).view(batch_size, -1, self.config['n_heads'], self.config['embedding_size']).transpose(-2, -3).transpose(-1, -2)

        if self.config['interaction_mask']:
            q_i = torch.einsum('bhne,bn->bhne', q_i, mask_i)
            k_i = torch.einsum('bhen,bn->bhen', k_i, mask_i)
            q_j = torch.einsum('bhne,bn->bhne', q_j, mask_j)
            k_j = torch.einsum('bhen,bn->bhen', k_j, mask_j)

        a_i = torch.matmul(q_i, k_j)
        a_i *= self.scale

        a_j = torch.matmul(q_j, k_i).transpose(-1, -2)
        a_j *= self.scale

        a = torch.cat([a_i, a_j], dim=1)

        return a
