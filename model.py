import torch
import torch.nn as nn
from attention_module import AdditiveAttention

class BiLSTM_CNN_Attention(nn.Module):
    def __init__(self, input_dim, meta_dim, hidden_dim=64, cnn_out_channels=64, kernel_size=3, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.conv1d = nn.Conv1d(in_channels=2*hidden_dim, out_channels=cnn_out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.attn = AdditiveAttention(cnn_out_channels)
        self.meta_fc = nn.Linear(meta_dim, cnn_out_channels)
        self.classifier = nn.Linear(cnn_out_channels*2, output_dim)

    def forward(self, x_seq, x_meta):
        lstm_out, _ = self.lstm(x_seq)  # [B, T, 2*H]
        conv_in = lstm_out.permute(0, 2, 1)  # [B, 2H, T]mod
        cnn_out = self.conv1d(conv_in).permute(0, 2, 1)  # [B, T, C]
        attn_out, attn_weights = self.attn(cnn_out)
        meta_out = torch.relu(self.meta_fc(x_meta))
        concat = torch.cat([attn_out, meta_out], dim=1)
        output = self.classifier(concat)
        return output, attn_weights
