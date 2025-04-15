import torch

from models.NAGSL.ChannelAlignmentModule import ChannelAlignment
from models.NAGSL.SimCNNModule import SimCNN
from models.NAGSL.SimMatPooling import SimMatPooling


class SimMatLearning(torch.nn.Module):
    def __init__(self, config):
        super(SimMatLearning, self).__init__()
        self.config = config

        if self.config['channel_align']:
            self.channel_alignment = ChannelAlignment(self.config).to(self.config['device'])

        if self.config['sim_mat_learning_ablation']:
            self.sim_mat_pooling = SimMatPooling(self.config).to(self.config['device'])
        else:
            self.sim_CNN = SimCNN(self.config).to(self.config['device'])

    def forward(self, mat, mask_ij):
        if self.config['channel_align']:
            mat = self.channel_alignment(mat, mask_ij)

        if self.config['sim_mat_learning_ablation']:
            score = self.sim_mat_pooling(mat)
        else:
            score = self.sim_CNN(mat, mask_ij)

        return score
