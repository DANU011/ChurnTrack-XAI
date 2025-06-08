import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaScaledAdditiveAttention(nn.Module):
    def __init__(self, input_dim, meta_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, 1)
        self.meta_proj = nn.Linear(meta_dim, input_dim)

    def forward(self, inputs, meta_vec):
        # inputs: [B, T, C], meta_vec: [B, D]
        meta_scaled = self.meta_proj(meta_vec).unsqueeze(1)  # [B, 1, C]
        combined = inputs + meta_scaled  # broadcasting
        scores = self.fc2(torch.tanh(self.fc1(combined)))  # [B, T, 1]
        weights = F.softmax(scores, dim=1)  # [B, T, 1]
        context = (inputs * weights).sum(dim=1)  # [B, C]
        return context, weights
