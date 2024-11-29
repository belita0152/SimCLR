import numpy as np
import torch
import torch.nn as nn
from backbone import *
from utils import get_backbone_model
import torch.nn.functional as F


# pass through projection head
class ProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.layers = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features, False),
            nn.ReLu(),
            nn.Linear(self.hidden_features, self.out_features, False)
            )

    def forward(self, x):
        x = self.layers(x)
        return x


# loss function for a positive pair
class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.pairwise_similarity = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * batch_size):
            mask[i, batch_size*batch_size + i] = 0
            mask[batch_size * batch_size + i, i] = 0

         return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim_i_k = self.pairwise_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim_i_k, self.batch_size * self.batch_size)
        sim_j_i = torch.diag(sim_i_k, -self.batch_size * self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim_i_k[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


# SimCLR's main learning algorithm (including NT-Xent calculation)
class SimCLR(object):
    def __init__(self, backbone_name, backbone_parameter, projection_hidden, projection_size, NT_Xent):
        super().__init__()

        # x -> augmentation

        # pass through base encoder (EEGNet)
        self.backbone = get_backbone_model(model_name=backbone_name,
                                           parameters=backbone_parameter)

        # pass through projection head
        self.projection = ProjectionHead(in_features=self.backbone, # backbone의 output
                                         hidden_features=projection_hidden,
                                         out_features=projection_size)

        # calculate info_nce_loss
        self.loss = NT_Xent()


    def forward(self, features):



