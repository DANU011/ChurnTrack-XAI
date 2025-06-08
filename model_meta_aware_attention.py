import torch
import torch.nn as nn
from attention_module_meta_aware_attention import MetaScaledAdditiveAttention

class BiLSTM_CNN_Attention(nn.Module):
    def __init__(self, input_dim, meta_dim, hidden_dim=64, cnn_out_channels=64, kernel_size=3, output_dim=1):
        super().__init__()

        # 1. BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # 2. CNN: BiLSTM 출력 채널을 입력 채널로 받음
        self.conv1d = nn.Conv1d(
            in_channels=hidden_dim * 2,
            out_channels=cnn_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        # 3. Attention: meta 정보 반영
        self.attn = MetaScaledAdditiveAttention(
            input_dim=cnn_out_channels,
            meta_dim=meta_dim
        )

        # 4. Meta 정보 임베딩
        self.meta_fc = nn.Linear(meta_dim, cnn_out_channels)

        # 5. 최종 분류기
        self.classifier = nn.Linear(cnn_out_channels * 2, output_dim)

    def forward(self, x_seq, x_meta):
        # x_seq: [B, T, F] / x_meta: [B, D]

        # 1. LSTM → CNN
        lstm_out, _ = self.lstm(x_seq)                     # [B, T, H*2]
        conv_in = lstm_out.permute(0, 2, 1)                # [B, H*2, T]
        cnn_out = self.conv1d(conv_in).permute(0, 2, 1)    # [B, T, C]

        # 2. Attention (meta-aware)
        attn_out, attn_weights = self.attn(cnn_out, x_meta)  # [B, C], [B, T, 1]

        # 3. Meta projection
        meta_out = torch.relu(self.meta_fc(x_meta))        # [B, C]

        # 4. Concatenate & classify
        concat = torch.cat([attn_out, meta_out], dim=1)    # [B, C*2]
        logits = self.classifier(concat)                   # [B, 1]

        return logits, attn_weights, meta_out

