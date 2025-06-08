import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_module import AdditiveAttention

class BiLSTM_CNN_Attention(nn.Module):
    def __init__(self, input_dim, meta_dim, hidden_dim, cnn_out_channels, kernel_size):
        super(BiLSTM_CNN_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.cnn_out_channels = cnn_out_channels

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv1d = nn.Conv1d(
            in_channels=2 * hidden_dim,
            out_channels=cnn_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.attn = AdditiveAttention(cnn_out_channels)

        # Meta gate for feature-wise modulation
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim, cnn_out_channels),
            nn.ReLU(),
            nn.Linear(cnn_out_channels, cnn_out_channels),
            nn.Sigmoid()  # Gate vector [B, C]
        )

        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Binary classification
        )

    def forward(self, x_seq, x_meta):
        # 1. LSTM → [B, T, 2H]
        lstm_out, _ = self.lstm(x_seq)

        # 2. CNN → [B, C, T]
        cnn_in = lstm_out.transpose(1, 2)     # [B, 2H, T]
        cnn_out = self.conv1d(cnn_in)         # [B, C, T]

        # 3. Meta gate → [B, C, 1]
        meta_gate = self.meta_fc(x_meta).unsqueeze(-1)  # [B, C] → [B, C, 1]
        gated_cnn_out = cnn_out * meta_gate             # [B, C, T]

        # 4. Attention → [B, C]
        attn_out, attn_weights = self.attn(gated_cnn_out.transpose(1, 2))  # [B, T, C] → [B, C]

        # 5. Classifier
        out = self.classifier(attn_out)  # [B, 1]
        return out, attn_weights, meta_gate.squeeze(-1)  # [B, C] gate for logging
