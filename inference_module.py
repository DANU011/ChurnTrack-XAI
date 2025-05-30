import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model import BiLSTM_CNN_Attention
from dataloader import create_dataloaders


def visualize_attention(input_seq, attn_weights, user_id=None, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.heatmap(attn_weights.detach().cpu().numpy(), cmap='viridis', xticklabels=input_seq, yticklabels=['Attention'])
    plt.xlabel('Time Steps')
    plt.title(f'Attention Weights for User {user_id}' if user_id else 'Attention Weights')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def inference(model_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM_CNN_Attention().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataloader = create_dataloaders(data_path, batch_size=1, shuffle=False)

    for batch in dataloader:
        time_series, meta_features, label, user_id = batch
        time_series = time_series.to(device)
        meta_features = meta_features.to(device)

        with torch.no_grad():
            output, attn_weights = model(time_series, meta_features, return_attention=True)
            pred = torch.sigmoid(output).item()

        print(f"User {user_id.item()} Prediction: {pred:.4f} (True: {label.item()})")
        visualize_attention(range(time_series.shape[1]), attn_weights.squeeze(0), user_id.item())
