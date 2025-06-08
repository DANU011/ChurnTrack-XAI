# train_v1.py (또는 사용 중인 훈련 스크립트)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import create_stratified_dataloaders
from model import BiLSTM_CNN_Attention
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


# ① 파이썬 내장 random 모듈 시드 고정
random.seed(42)
# ② NumPy 시드 고정
np.random.seed(42)
# ③ PyTorch CPU 시드 고정
torch.manual_seed(42)
# ④ PyTorch GPU 시드 고정 (CUDA가 사용 가능한 경우)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)   # 멀티 GPU 환경이라면
# ⑤ PyTorch 연산 결과를 좀 더 결정론적으로 만들고 싶다면
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_model(config, time_series_df, meta_df, save_suffix):
    lambda_reg = config.get("meta_reg_lambda", 1e-4) # 정규화 계수
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ①-1 bool 타입 → float 타입으로 변환 (시계열 DataFrame 전체에 적용)
    bool_cols = time_series_df.select_dtypes(include=['bool']).columns
    time_series_df[bool_cols] = time_series_df[bool_cols].astype(float)

    # ①-2 시계열(input_seq)로 사용할 숫자형 칼럼명 추출 (존재 여부 체크하며 제외)
    ts_num_cols = time_series_df.select_dtypes(include=[np.number]).columns.tolist()
    ts_numeric_cols = [c for c in ts_num_cols if c not in ['month', 'customer_id']]
    print(f">>> ts_numeric_cols (총 {len(ts_numeric_cols)}개): {ts_numeric_cols}")

    # ②-1 bool 타입 → float 타입으로 변환 (메타 정보 DataFrame 전체에 적용)
    bool_cols_meta = meta_df.select_dtypes(include=['bool']).columns
    meta_df[bool_cols_meta] = meta_df[bool_cols_meta].astype(float)

    # ②-2 메타(input_meta)로 사용할 숫자형 칼럼명 추출 (존재 여부 체크하며 제외)
    meta_num_cols = meta_df.select_dtypes(include=[np.number]).columns.tolist()
    meta_numeric_cols = [c for c in meta_num_cols if c not in ['churn', 'customer_id']]
    print(f">>> meta_numeric_cols (총 {len(meta_numeric_cols)}개): {meta_numeric_cols}")

    # 1) DataLoader: (cid, x_seq, x_meta, y) 형식으로 반환됨
    train_loader, val_loader, test_loader, meta_info = create_stratified_dataloaders(
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
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=(0.85, 0.999),
        eps=1e-7
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    best_auc = 0
    save_path = ""
    shap_path = ""
    result_path = ""
    attention_path = ""
    attention_plot_path = ""  # 추가: Attention 시각화 결과 경로
    history = []

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0

        shap_x_seq = []
        shap_x_meta = []
        shap_y = []

        for cid, x_seq, x_meta, y in tqdm(train_loader):
            x_seq, x_meta, y = x_seq.to(device), x_meta.to(device), y.to(device).float()
            optimizer.zero_grad()
            # output, _ = model(x_seq, x_meta)
            # loss = criterion(output.squeeze(), y)
            output, _, meta_out = model(x_seq, x_meta)
            loss_main = criterion(output.squeeze(), y)
            loss_reg = torch.norm(meta_out, p=2)  # p=1로 바꾸면 L1 정규화
            loss = loss_main + lambda_reg * loss_reg

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"[Epoch {epoch + 1}] Loss: {epoch_loss / len(train_loader):.4f}")

        model.eval()
        y_true, y_pred = [], []
        customer_ids = []       # ← 추가: 검증 시각에 고객 ID 수집
        attention_list = []

        with torch.no_grad():
            for cid, x_seq, x_meta, y in val_loader:
                x_seq, x_meta, y = x_seq.to(device), x_meta.to(device), y.to(device).float()
                output, attention, _ = model(x_seq, x_meta)

                cid_list = cid.tolist()  # CPU상태의 텐서를 리스트로 변환
                customer_ids.extend(cid_list)

                # raw logit → sigmoid 확률로 변환
                probs = torch.sigmoid(output.squeeze())  # shape: [batch_size]

                y_true.extend(y.tolist())
                y_pred.extend(probs.squeeze().cpu().tolist())

                shap_x_seq.append(x_seq.cpu().numpy())
                shap_x_meta.append(x_meta.cpu().numpy())
                shap_y.extend(y.cpu().numpy())

                # attention: [batch_size, seq_len, 1] 혹은 [batch_size, seq_len]
                # numpy로 변환해 리스트에 저장
                attention_np = attention.cpu().numpy()
                if attention_np.ndim == 3 and attention_np.shape[-1] == 1:
                    attention_np = attention_np.squeeze(-1)  # [B, seq_len]
                attention_list.append(attention_np)

        # Flatten prediction 결과
        y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
        auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
        f1 = f1_score(y_true, y_pred_binary) if len(set(y_true)) > 1 else 0.0
        acc = accuracy_score(y_true, y_pred_binary)
        print(f"[Validation] AUC: {auc:.4f}, F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")

        scheduler.step(auc)
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

            # 1) SHAP 입력 저장
            np.savez_compressed(
                f"shap_input_{save_suffix}.npz",
                x_seq=np.concatenate(shap_x_seq, axis=0),
                x_meta=np.concatenate(shap_x_meta, axis=0),
                y=np.array(shap_y),
                seq_feature_names=np.array(ts_numeric_cols),
                meta_feature_names=np.array(meta_numeric_cols)
            )
            shap_path = f"shap_input_{save_suffix}.npz"
            print(f"[Saved] SHAP input saved with feature names: {shap_path}")



            # 2) 검증 예측 결과에 customer_id 추가
            result_path = f"val_preds_{save_suffix}.csv"
            with open(result_path, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['customer_id', 'y_true', 'y_pred'])
                for cid_val, true_val, pred_val in zip(customer_ids, y_true, y_pred):
                    writer.writerow([cid_val, true_val, pred_val])
            print(f"[Saved] Validation predictions (with customer_id) to: {result_path}")

            # 3) Attention 저장
            attention_path = f"attention_{save_suffix}.npy"
            # attention_list: 리스트 안에 배치별 (batch_size, seq_len) 배열이 담겨 있음
            # 따라서 concatenate 하면 shape = (num_val_samples, seq_len)
            attention_all = np.concatenate(attention_list, axis=0)
            np.save(attention_path, attention_all)
            print(f"[Saved] Attention weights saved to: {attention_path}")

            # 4) Attention 시각화 (히트맵)
            # (1) 시퀀스 길이
            seq_len = attention_all.shape[1]

            # (2) validation 샘플 전체에서 step별 평균 attention 계산
            avg_attention = attention_all.mean(axis=0)  # shape = (seq_len,)

            # (3) 히트맵 형태(1×seq_len)로 그리기
            plt.figure(figsize=(seq_len / 2, 2))  # 가로 길이를 seq_len에 맞춰 넓게
            plt.imshow(avg_attention[np.newaxis, :], aspect='auto', cmap='viridis')
            plt.colorbar(label='Attention weight (avg)')
            plt.xticks(ticks=np.arange(seq_len), labels=np.arange(1, seq_len + 1))
            plt.yticks([])  # y축 레이블은 필요없음
            plt.xlabel("Time step (1 to seq_len)")
            plt.title("Average Attention over Validation Set")

            attention_plot_path = f"attention_heatmap_{save_suffix}.png"
            plt.tight_layout()
            plt.savefig(attention_plot_path, dpi=300)
            plt.close()
            print(f"[Saved] Attention heatmap to: {attention_plot_path}")

            # test
            best_model = BiLSTM_CNN_Attention(
                input_dim=meta_info['num_features'],
                meta_dim=meta_info['meta_dim'],
                hidden_dim=config['hidden_dim'],
                cnn_out_channels=config['cnn_out_channels'],
                kernel_size=config['kernel_size']
            ).to(device)
            best_model.load_state_dict(torch.load(save_path, map_location=device))
            best_model.eval()

            test_y_true, test_y_pred = [], []
            test_customer_ids = []

            # SHAP용 입력 저장 (test set 기준)
            shap_x_seq_test = []
            shap_x_meta_test = []
            shap_y_test = []

            with torch.no_grad():
                for cid_t, x_seq_t, x_meta_t, y_t in test_loader:
                    x_seq_t, x_meta_t = x_seq_t.to(device), x_meta_t.to(device)
                    logits_t, _, _ = best_model(x_seq_t, x_meta_t)
                    probs_t = torch.sigmoid(logits_t.squeeze()).cpu().tolist()

                    test_customer_ids.extend(cid_t.tolist())
                    test_y_true.extend(y_t.tolist())
                    test_y_pred.extend(probs_t)

                    shap_x_seq_test.append(x_seq_t.cpu().numpy())
                    shap_x_meta_test.append(x_meta_t.cpu().numpy())
                    shap_y_test.extend(y_t.cpu().numpy())

            shap_test_path = f"shap_input_test_{save_suffix}.npz"
            np.savez_compressed(
                shap_test_path,
                x_seq=np.concatenate(shap_x_seq_test, axis=0),
                x_meta=np.concatenate(shap_x_meta_test, axis=0),
                y=np.array(shap_y_test),
                seq_feature_names=np.array(ts_numeric_cols),
                meta_feature_names=np.array(meta_numeric_cols)
            )
            print(f"[Saved] SHAP test input saved to: {shap_test_path}")

            # 이진 분류 지표 계산
            test_y_pred_binary = [1 if p > 0.5 else 0 for p in test_y_pred]
            test_auc = roc_auc_score(test_y_true, test_y_pred) if len(set(test_y_true)) > 1 else 0.0
            test_f1 = f1_score(test_y_true, test_y_pred_binary) if len(set(test_y_true)) > 1 else 0.0
            test_acc = accuracy_score(test_y_true, test_y_pred_binary)
            print(f"[Test] AUC: {test_auc:.4f}, F1 Score: {test_f1:.4f}, Accuracy: {test_acc:.4f}")

            test_result_path = f"test_preds_{save_suffix}.csv"
            with open(test_result_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(['customer_id', 'y_true', 'y_pred'])
                for cid_val, true_val, pred_val in zip(test_customer_ids, test_y_true, test_y_pred):
                    writer.writerow([cid_val, true_val, pred_val])
            print(f"[Saved] Test predictions (with customer_id) to: {test_result_path}")
            # test 끝

    log_path = f"train_log_{save_suffix}.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"[Log Saved] Performance log saved to {log_path}")

    return {
        'best_model_path': save_path,
        'best_auc': best_auc,
        'log_path': log_path,
        'val_pred_path': result_path,
        'shap_input_path': shap_path,
        'attention_path': attention_path,
        'attention_plot_path': attention_plot_path  # 추가: 시각화 결과 경로
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
        'kernel_size': 3,
        'meta_reg_lambda': 1e-4
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
