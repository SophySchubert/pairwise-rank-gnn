"""Entire Code from https://github.com/AlbertTan404/NA-GSL/tree/main accessed on 2 June 2025"""
import torch
from models.NAGSL.EmbeddingLearning import GCNTransformerEncoder
from models.NAGSL.EmbeddingInteraction import CrossTransformer
from models.NAGSL.SimMatLearning import SimMatLearning

class NAGSLNet(torch.nn.Module):
    def __init__(self, config):
        super(NAGSLNet, self).__init__()
        self.config = config

        if self.config['share_qk']:
            q = torch.nn.Linear(self.config['embedding_size'], self.config['embedding_size'] * self.config['n_heads'], bias=self.config['msa_bias'])
            k = torch.nn.Linear(self.config['embedding_size'], self.config['embedding_size'] * self.config['n_heads'], bias=self.config['msa_bias'])
        else:
            q = k = None

        self.embedding_learning = GCNTransformerEncoder(self.config, q, k).to(config['device'])

        self.embedding_interaction = CrossTransformer(self.config, q, k).to(config['device'])

        self.sim_mat_learning = SimMatLearning(self.config).to(config['device'])

    def forward(self, data):

        x_0 = data['g0']['x']
        adj_0 = data['g0']['adj']
        mask_0 = data['g0']['mask']
        dist_0 = data['g0']['dist']
        x_1 = data['g1']['x']
        adj_1 = data['g1']['adj']
        mask_1 = data['g1']['mask']
        dist_1 = data['g1']['dist']

        embeddings_0 = self.embedding_learning(x_0, adj_0, mask_0, dist_0)
        embeddings_1 = self.embedding_learning(x_1, adj_1, mask_1, dist_1)

        if self.config['encoder_mask'] or self.config['interaction_mask'] or self.config['align_mask'] or self.config['cnn_mask']:
            mask_ij = torch.einsum('ij,ik->ijk', mask_0, mask_1)
        else:
            mask_ij = None

        sim_mat = self.embedding_interaction(embeddings_0, mask_0, embeddings_1, mask_1, mask_ij)
        score = self.sim_mat_learning(sim_mat, mask_ij)

        return score
