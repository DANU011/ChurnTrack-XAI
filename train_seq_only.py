import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader_seq_only import create_stratified_dataloaders
from model_seq_only import BiLSTM_CNN_Attention
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tqdm import tqdm
import os
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random

font_path = "fonts/NanumGothicCoding-2.0/나눔고딕코딩.ttf"
if not os.path.exists(font_path):
    font_path = "/home/danu/deep/fonts/NanumGothicCoding-2.0/나눔고딕코딩.ttf"
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_model(config, time_series_df, save_suffix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bool_cols = time_series_df.select_dtypes(include=['bool']).columns
    time_series_df[bool_cols] = time_series_df[bool_cols].astype(float)

    ts_num_cols = time_series_df.select_dtypes(include=[np.number]).columns.tolist()
    ts_numeric_cols = [c for c in ts_num_cols if c not in ['month', 'customer_id']]
    print(f">>> ts_numeric_cols ({len(ts_numeric_cols)}): {ts_numeric_cols}")

    train_loader, val_loader, test_loader, meta_info = create_stratified_dataloaders(
        time_series_df=time_series_df,
        window_size=config['window_size'],
        cache_path=f"cached_dataset_{save_suffix}.npz",
        batch_size=config.get('batch_size', 64)
    )

    model = BiLSTM_CNN_Attention(
        input_dim=meta_info['num_features'],
        hidden_dim=config['hidden_dim'],
        cnn_out_channels=config['cnn_out_channels'],
        kernel_size=config['kernel_size']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6)

    best_auc = 0
    history = []

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0

        for _, x_seq, y in tqdm(train_loader):
            x_seq, y = x_seq.to(device), y.to(device).float()
            optimizer.zero_grad()
            output, _ = model(x_seq)
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"[Epoch {epoch + 1}] Loss: {epoch_loss / len(train_loader):.4f}")

        model.eval()
        y_true, y_pred = [], []
        shap_x_seq, shap_y = [], []
        attention_list = []
        with torch.no_grad():
            for _, x_seq, y in val_loader:
                x_seq, y = x_seq.to(device), y.to(device).float()
                output, attention = model(x_seq)
                probs = torch.sigmoid(output.squeeze())
                y_true.extend(y.tolist())
                y_pred.extend(probs.cpu().tolist())
                shap_x_seq.append(x_seq.cpu().numpy())
                shap_y.extend(y.cpu().numpy())
                attention_np = attention.cpu().numpy()
                if attention_np.ndim == 3 and attention_np.shape[-1] == 1:
                    attention_np = attention_np.squeeze(-1)
                attention_list.append(attention_np)

        auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
        f1 = f1_score(y_true, [1 if p > 0.5 else 0 for p in y_pred]) if len(set(y_true)) > 1 else 0.0
        acc = accuracy_score(y_true, [1 if p > 0.5 else 0 for p in y_pred])
        print(f"[Validation] AUC: {auc:.4f}, F1: {f1:.4f}, ACC: {acc:.4f}")

        scheduler.step(auc)
        for param_group in optimizer.param_groups:
            print(f"Current LR: {param_group['lr']:.6f}")

        history.append({'epoch': epoch + 1, 'loss': epoch_loss / len(train_loader), 'auc': auc, 'f1': f1, 'acc': acc})

        if auc > best_auc:
            best_auc = auc
            save_path = config['save_path'].replace('.pth', f'_{save_suffix}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"[Saved] Best model with AUC: {best_auc:.4f} → {save_path}")

            np.savez_compressed(f"shap_input_{save_suffix}.npz",
                                x_seq=np.concatenate(shap_x_seq, axis=0),
                                y=np.array(shap_y),
                                seq_feature_names=np.array(ts_numeric_cols))

            attention_all = np.concatenate(attention_list, axis=0)
            np.save(f"attention_{save_suffix}.npy", attention_all)

            avg_attention = attention_all.mean(axis=0)
            plt.figure(figsize=(avg_attention.shape[0]/2, 2))
            plt.imshow(avg_attention[np.newaxis, :], aspect='auto', cmap='viridis')
            plt.colorbar(label='Attention Weight (avg)')
            plt.xticks(ticks=np.arange(avg_attention.shape[0]), labels=np.arange(1, avg_attention.shape[0]+1))
            plt.yticks([])
            plt.xlabel("Time Step")
            plt.title("Average Attention Heatmap")
            plt.tight_layout()
            plt.savefig(f"attention_heatmap_{save_suffix}.png", dpi=300)
            plt.close()

            best_model = BiLSTM_CNN_Attention(
                input_dim=meta_info['num_features'],
                hidden_dim=config['hidden_dim'],
                cnn_out_channels=config['cnn_out_channels'],
                kernel_size=config['kernel_size']
            ).to(device)
            best_model.load_state_dict(torch.load(save_path))
            best_model.eval()

            test_y_true, test_y_pred = [], []
            shap_x_seq_test, shap_y_test = [], []
            with torch.no_grad():
                for _, x_seq, y in test_loader:
                    x_seq = x_seq.to(device)
                    logits, _ = best_model(x_seq)
                    probs = torch.sigmoid(logits.squeeze()).cpu().tolist()
                    test_y_true.extend(y.tolist())
                    test_y_pred.extend(probs)
                    shap_x_seq_test.append(x_seq.cpu().numpy())
                    shap_y_test.extend(y.cpu().numpy())

            np.savez_compressed(
                f"shap_input_test_{save_suffix}.npz",
                x_seq=np.concatenate(shap_x_seq_test, axis=0),
                y=np.array(shap_y_test),
                seq_feature_names=np.array(ts_numeric_cols)
            )
            print(f"[Saved] SHAP test input saved to: shap_input_test_{save_suffix}.npz")

            with open(f"test_preds_{save_suffix}.csv", "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['y_true', 'y_pred'])
                for yt, yp in zip(test_y_true, test_y_pred):
                    writer.writerow([yt, yp])

    log_path = f"train_log_{save_suffix}.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"[Log Saved] → {log_path}")

    return {
        'best_model_path': save_path,
        'best_auc': best_auc,
        'log_path': log_path,
        'val_pred_path': f"val_preds_{save_suffix}.csv",
        'shap_input_path': f"shap_input_{save_suffix}.npz",
        'attention_path': f"attention_{save_suffix}.npy",
        'attention_plot_path': f"attention_heatmap_{save_suffix}.png",
        'test_pred_path': f"test_preds_{save_suffix}.csv",
        'shap_test_input_path': f"shap_input_test_{save_suffix}.npz"
    }

if __name__ == "__main__":
    config = {
        'time_data_path': 'time_sample_filtered.csv',
        'save_path': 'best_churn_model.pth',
        'window_size': 5,
        'stride': 1,
        'hidden_dim': 64,
        'lr': 1e-2,
        'epochs': 200,
        'cnn_out_channels': 64,
        'kernel_size': 3
    }

    df = pd.read_csv(config['time_data_path'])
    df = df.sort_values(['customer_id', 'month'])
    df = df.groupby('customer_id', group_keys=False).apply(lambda grp: grp.ffill()).fillna(0).reset_index(drop=True)

    label_df = df[['customer_id', 'churn']].drop_duplicates()
    start_month = 201807
    end_month = start_month + 4
    label_month = start_month + 5

    ts_df = df[(df['month'] >= start_month) & (df['month'] <= end_month)].copy()
    ts_df = ts_df.groupby('customer_id', group_keys=False).apply(lambda grp: grp.ffill()).fillna(0).reset_index(drop=True)

    churn_1 = label_df[label_df['churn'] == 1]
    churn_0 = label_df[label_df['churn'] == 0].sample(n=len(churn_1), random_state=42)
    filtered_ids = pd.concat([churn_1, churn_0])['customer_id'].unique()
    time_series_df = ts_df[ts_df['customer_id'].isin(filtered_ids)].copy()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_suffix = f"{start_month}_{label_month}_{timestamp}"

    outputs = train_model(config, time_series_df, save_suffix)

    print("Returned paths:")
    for key, path in outputs.items():
        print(f"{key}: {path}")
