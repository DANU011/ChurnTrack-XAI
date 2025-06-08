import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        scores = self.fc2(torch.tanh(self.fc1(inputs)))
        weights = F.softmax(scores, dim=1)
        weighted = inputs * weights
        context = weighted.sum(dim=1)
        return context, weights
