import torch
import torch.nn as nn
from backbone import *
import torch.nn.functional as F


# pass through projection head
class ProjectionHead(nn.Module):  # learnable nonlinear transformation between representation and contrastive loss
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


class NTXent(object):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='sum')  # softmax + CrossEntropy

    def forward(self, z_i, z_j):
        tensor = torch.arange(self.batch_size)
        class_labels = torch.cat([tensor for i in range(2)], dim=0)
        class_labels = (class_labels.unsqueeze(0) == class_labels.unsqueeze((1))).float()

        features = torch.cat((z_i, z_j), dim=0)
        features = F.normalize(features, dim=-1)
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(class_labels.shape[0], dtype=torch.bool)

        class_labels = class_labels[~mask].view(class_labels.shape[0], -1)

        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[class_labels.bool()].view(class_labels.shape[0], -1)
        negatives = similarity_matrix[~class_labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        loss = self.criterion(logits, labels)

        return (logits, labels), loss
