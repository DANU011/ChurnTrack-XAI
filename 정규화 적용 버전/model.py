# import torch
# import torch.nn as nn
# from attention_module import AdditiveAttention
#
# class BiLSTM_CNN_Attention(nn.Module):
#     def __init__(self, input_dim, meta_dim, hidden_dim=64, cnn_out_channels=64, kernel_size=3, output_dim=1):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
#         self.conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=cnn_out_channels, kernel_size=kernel_size, padding=kernel_size//2)
#         self.attn = AdditiveAttention(cnn_out_channels)
#         self.meta_fc = nn.Linear(meta_dim, cnn_out_channels)
#         self.classifier = nn.Linear(cnn_out_channels*2, output_dim)
#
#     def forward(self, x_seq, x_meta):
#         lstm_out, _ = self.lstm(x_seq)  # [B, T, H]
#         conv_in = lstm_out.permute(0, 2, 1)  # [B, H, T]
#         cnn_out = self.conv1d(conv_in).permute(0, 2, 1)  # [B, T, C]
#         attn_out, attn_weights = self.attn(cnn_out)
#         meta_out = torch.relu(self.meta_fc(x_meta))
#         concat = torch.cat([attn_out, meta_out], dim=1)
#         logits = self.classifier(concat)
#         return logits, attn_weights

import torch
import torch.nn as nn
from attention_module import AdditiveAttention

class BiLSTM_CNN_Attention(nn.Module):
    def __init__(self, input_dim, meta_dim, hidden_dim=64, cnn_out_channels=64, kernel_size=3, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True  # ← 단방향 → 양방향으로 변경
        )
        self.conv1d = nn.Conv1d(
            in_channels=hidden_dim * 2,  # ← BiLSTM이므로 hidden_dim × 2
            out_channels=cnn_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.attn = AdditiveAttention(cnn_out_channels)
        self.meta_fc = nn.Linear(meta_dim, cnn_out_channels)
        self.classifier = nn.Linear(cnn_out_channels * 2, output_dim)

    def forward(self, x_seq, x_meta):
        lstm_out, _ = self.lstm(x_seq)             # [B, T, H*2] ← BiLSTM 출력
        conv_in = lstm_out.permute(0, 2, 1)        # [B, H*2, T]
        cnn_out = self.conv1d(conv_in).permute(0, 2, 1)  # [B, T, C]
        attn_out, attn_weights = self.attn(cnn_out)      # [B, C]
        meta_out = torch.relu(self.meta_fc(x_meta))      # [B, C]
        concat = torch.cat([attn_out, meta_out], dim=1)  # [B, C*2]
        logits = self.classifier(concat)                 # [B, 1]
        return logits, attn_weights, meta_out

