import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import create_stratified_dataloaders
from model import BiLSTM_CNN_Attention
# from dataloader import create_dataloaders
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tqdm import tqdm
import os
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime

def train_model(config, time_series_df, meta_df, save_suffix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # train_loader, val_loader, _, meta_info = create_dataloaders(
    #     time_series_df=time_series_df,
    #     meta_df=meta_df,
    #     batch_size=config.get('batch_size', 64),
    #     window_size=config['window_size'],
    #     cache_path=f"cached_dataset_{save_suffix}.npz"  # Added support for caching
    # )

    # train_loader, val_loader, _, meta_info = create_stratified_dataloaders(
    #     cache_path=f"cached_dataset_201807_201812.npz", # 캐시 파일 고정
    #     batch_size=config.get('batch_size', 64)
    # )

    train_loader, val_loader, _, meta_info = create_stratified_dataloaders(
        time_series_df=time_series_df,
        meta_df=meta_df,
        window_size=config['window_size'],
        cache_path=f"cached_dataset_{save_suffix}.npz",
        batch_size=config.get('batch_size', 64)
    )

    model = BiLSTM_CNN_Attention(
        input_dim=meta_info['num_features'],
        meta_dim=meta_info['meta_dim'],
        hidden_dim=config['hidden_dim'],
        cnn_out_channels=config['cnn_out_channels'],
        kernel_size=config['kernel_size']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # adam 설정 변경
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=(0.85, 0.999),  # 모멘텀 조정
        eps=1e-7  # 수치 안정성 향상
    )
    # 스케쥴러
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # AUC 높을수록 좋은 경우
        factor=0.5,  # lr 감소 비율
        patience=5,  # 5번 연속 향상 없으면 감소
        verbose=True,  # 출력 로그 활성화
        min_lr = 1e-6
    )

    best_auc = 0
    save_path = ""
    shap_path = ""
    result_path = ""
    attention_path = ""
    history = []

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0

        shap_x_seq = []
        shap_x_meta = []
        shap_y = []

        for x_seq, x_meta, y in tqdm(train_loader):
            x_seq, x_meta, y = x_seq.to(device), x_meta.to(device), y.to(device).float()
            optimizer.zero_grad()
            output, _ = model(x_seq, x_meta)
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {epoch_loss / len(train_loader):.4f}")

        model.eval()
        y_true, y_pred = [], []
        attention_list = []

        with torch.no_grad():
            for x_seq, x_meta, y in val_loader:
                x_seq, x_meta = x_seq.to(device), x_meta.to(device)
                output, attention = model(x_seq, x_meta)

                # Attention weight 확인
                # print("Attention mean (val):", attention.mean().item())

                y_true.extend(y.tolist())
                y_pred.extend(output.squeeze().cpu().tolist())

                shap_x_seq.append(x_seq.cpu().numpy())
                shap_x_meta.append(x_meta.cpu().numpy())
                shap_y.extend(y.cpu().numpy())

                attention_list.append(attention.cpu().numpy())

        y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
        auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
        f1 = f1_score(y_true, y_pred_binary) if len(set(y_true)) > 1 else 0.0
        acc = accuracy_score(y_true, y_pred_binary)
        print(f"[Validation] AUC: {auc:.4f}, F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")
        # validation AUC 기준으로 스케줄러 작동
        scheduler.step(auc)
        # 현재 lr
        for param_group in optimizer.param_groups:
            print(f"Current LR: {param_group['lr']:.6f}")

        history.append({
            'epoch': epoch + 1,
            'loss': epoch_loss / len(train_loader),
            'auc': auc,
            'f1': f1,
            'accuracy': acc
        })

        if auc > best_auc:
            best_auc = auc
            save_path = config['save_path'].replace('.pth', f'_{save_suffix}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"[Saved] Best model saved with AUC: {best_auc:.4f} → {save_path}")

            np.savez_compressed(
                f"shap_input_{save_suffix}.npz",
                x_seq=np.concatenate(shap_x_seq, axis=0),
                x_meta=np.concatenate(shap_x_meta, axis=0),
                y=np.array(shap_y)
            )
            shap_path = f"shap_input_{save_suffix}.npz"
            print(f"[Saved] SHAP input saved: {shap_path}")

            result_path = f"val_preds_{save_suffix}.csv"
            with open(result_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['y_true', 'y_pred'])
                writer.writerows(zip(y_true, y_pred))
            print(f"[Saved] Validation predictions to: {result_path}")

            attention_path = f"attention_{save_suffix}.npy"
            np.save(attention_path, np.concatenate(attention_list, axis=0))
            print(f"[Saved] Attention weights saved to: {attention_path}")

    log_path = f"train_log_{save_suffix}.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"[Log Saved] Performance log saved to {log_path}")

    return {
        'best_model_path': save_path,
        'best_auc': best_auc,
        'log_path': log_path,
        'val_pred_path': result_path,
        'shap_input_path': shap_path,
        'attention_path': attention_path
    }

if __name__ == "__main__":
    config = {
        'time_data_path': 'time_sample_filtered.csv',
        'meta_data_path': 'meta_merged_balanced.csv',
        'save_path': 'best_churn_model.pth',
        'window_size': 5,
        'stride': 1,
        'hidden_dim': 64,
        'lr': 1e-2,
        'epochs': 150,
        'cnn_out_channels': 64,
        'kernel_size': 3
    }

    # 1) 파일 로드
    df = pd.read_csv(config['time_data_path'])
    meta_all = pd.read_csv(config['meta_data_path'])

    # 2) Time-series 결측치 처리
    df = df.sort_values(['customer_id', 'month'])
    df = (
        df.groupby('customer_id', group_keys=False)
        .apply(lambda grp: grp.ffill())
        .fillna(0)
        .reset_index(drop=True)
    )

    # 3) Meta-data 결측치 처리
    num_meta_cols = meta_all.select_dtypes(include=[np.number]).columns
    meta_all[num_meta_cols] = meta_all[num_meta_cols].fillna(meta_all[num_meta_cols].median())

    for start_month in [201807]:
        end_month   = start_month + 4
        label_month = start_month + 5

        print(f"\nTraining on window: {start_month} ~ {end_month}, Predicting {label_month}")

        # 4) 시계열 슬라이싱 및 결측치 처리
        ts_df = df[(df['month'] >= start_month) & (df['month'] <= end_month)].copy()
        ts_df = (
            ts_df.sort_values(['customer_id', 'month'])
            .groupby('customer_id', group_keys=False)
            .apply(lambda grp: grp.ffill())
            .fillna(0)
            .reset_index(drop=True)
        )
        time_series_df = ts_df.drop(columns=['churn'], errors='ignore')

        # 5) 메타+라벨 병합
        label_df = meta_all[['customer_id', 'churn']].drop_duplicates()
        meta_feats = meta_all.drop(columns=['churn']).drop_duplicates()
        meta_df = pd.merge(label_df, meta_feats, on='customer_id', how='inner')

        # 5.5) 이탈/비이탈 1:1 비율로 언더샘플링
        churn_1 = meta_df[meta_df['churn'] == 1]
        churn_0 = meta_df[meta_df['churn'] == 0].sample(n=len(churn_1), random_state=42)

        meta_df = pd.concat([churn_1, churn_0], axis=0).sample(frac=1, random_state=42)
        time_series_df = time_series_df[time_series_df['customer_id'].isin(meta_df['customer_id'])]

        # 6) 학습 실행
        # 실행 시각을 기준으로 고유한 suffix 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_suffix = f"{start_month}_{label_month}_{timestamp}"

        outputs = train_model(
            config,
            time_series_df,
            meta_df,
            save_suffix=save_suffix
        )

        print("Returned paths:")
        for key, path in outputs.items():
            print(f"{key}: {path}")
