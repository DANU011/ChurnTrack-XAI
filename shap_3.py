import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from model import BiLSTM_CNN_Attention
import os

# ───────────────────────────────────────────────────────────────────────────────
# 1) 로컬 폰트 파일 경로 지정 (실제 경로로 수정)
font_path = "fonts/NanumGothicCoding-2.0/나눔고딕코딩.ttf"
if not os.path.exists(font_path):
    font_path = "/home/danu/deep/fonts/NanumGothicCoding-2.0/나눔고딕코딩.ttf"

# 2) 폰트를 Matplotlib에 등록
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False
# ───────────────────────────────────────────────────────────────────────────────

# 1. NPZ 파일 로드
data = np.load("shap_input_201807_201812_20250602_040642.npz")
print("Available keys in npz:", data.files)

# 메타/시퀀스 피처명 불러오기
if "meta_feature_names" in data.files:
    meta_feature_names = data["meta_feature_names"].tolist()
else:
    meta_feature_names = [f"meta_{i}" for i in range(data["x_meta"].shape[1])]
    print("meta_feature_names not found—using default names.")

if "seq_feature_names" in data.files:
    seq_feature_names = data["seq_feature_names"].tolist()
else:
    seq_feature_names = [f"seq_{i}" for i in range(data["x_seq"].shape[2])]
    print("seq_feature_names not found—using default names.")

# x_seq, x_meta, y 로드
x_seq  = torch.from_numpy(data["x_seq"]).float()   # (N, seq_len, input_dim)
x_meta = torch.from_numpy(data["x_meta"]).float()  # (N, meta_dim)
y      = torch.from_numpy(data["y"]).float().squeeze()

input_dim = x_seq.shape[2]
meta_dim  = x_meta.shape[1]
seq_len   = x_seq.shape[1]

print("x_seq.shape:", x_seq.shape)
print("x_meta.shape:", x_meta.shape)
print("input_dim:", input_dim)
print("meta_dim:", meta_dim)
print("seq_len:", seq_len)

# 2. 모델 로딩
model = BiLSTM_CNN_Attention(
    input_dim=input_dim,
    meta_dim=meta_dim,
    hidden_dim=64,
    cnn_out_channels=64,
    kernel_size=3
)
model.load_state_dict(torch.load("best_churn_model_201807_201812_20250602_040642.pth", map_location="cpu"))
model.eval()

# 3. model_fn 정의
def model_fn(input_array):
    batch_size = input_array.shape[0]
    arr = torch.from_numpy(input_array).float()

    # 시계열 부분 분리
    seq_part = arr[:, : seq_len * input_dim]                      # (batch_size, seq_len*input_dim)
    x_seq_in = seq_part.reshape(batch_size, seq_len, input_dim)   # (batch_size, seq_len, input_dim)

    # 메타 부분 분리
    x_meta_in = arr[:, seq_len * input_dim :]                      # (batch_size, meta_dim)

    with torch.no_grad():
        output, _ = model(x_seq_in, x_meta_in)                     # (batch_size, 1) 또는 (batch_size,)
        logits = output.view(-1)                                   # (batch_size,)
        probs  = torch.sigmoid(logits)                             # (batch_size,)
    return probs.cpu().numpy()                                      # (batch_size,)

# 4. 전체 예측 → “실제 churn이고 모델도 churn 예측” 샘플만 선택
model_input = torch.cat([x_seq.reshape(len(x_seq), -1), x_meta], dim=1)  # (N, seq_len*input_dim + meta_dim)
preds = model_fn(model_input.numpy())                                    # (N,)

# ───────────────────────────────────────────────────────────────────────────────
# 5. 두 구간으로 나눠서 “실제 y==1 & 예측확률 in 구간” 마스크 생성
mask1 = (y == 1) & ((preds >= 0.5) & (preds < 0.8))   # 0.5 ≤ preds < 0.8
mask2 = (y == 1) & (preds >= 0.8)                     # preds ≥ 0.8

print("Group1 (0.5 ≤ preds < 0.8) count:", mask1.sum().item())
print("Group2 (preds ≥ 0.8) count:", mask2.sum().item())

# 6. 그룹별 반복: 대표 입력(평균) → SHAP 계산 → summary_plot 저장
for group_idx, mask in enumerate([mask1, mask2], start=1):
    selected_seq  = x_seq[mask]   # (M, seq_len, input_dim)
    selected_meta = x_meta[mask]  # (M, meta_dim)

    if selected_seq.size(0) == 0:
        print(f"Group{group_idx}에 해당하는 샘플이 없습니다. 건너뜁니다.")
        continue

    # (A) 그룹별 대표 입력 생성
    avg_seq  = selected_seq.mean(dim=0)   # (seq_len, input_dim)
    avg_meta = selected_meta.mean(dim=0)  # (meta_dim,)

    avg_input = torch.cat([avg_seq.reshape(-1), avg_meta], dim=0).unsqueeze(0).numpy()  # (1, 310+57)

    # (B) SHAP 배경(reference) 준비: 항상 처음 100개 샘플 사용
    bg_n = min(100, x_seq.shape[0])
    background = np.concatenate([
        x_seq[:bg_n].numpy().reshape(bg_n, -1),
        x_meta[:bg_n].numpy()
    ], axis=1)  # (bg_n, 367)

    # (C) SHAP 계산
    explainer = shap.KernelExplainer(model_fn, background)
    print(f"Computing SHAP for Group{group_idx} averaged churn sample...")
    shap_vals_for_avg = explainer.shap_values(avg_input, nsamples=100)[0]  # (367,)

    # (D) 시계열/메타로 분할 후 Summary Plot 생성
    seq_feat_len  = input_dim * seq_len   # 310
    meta_feat_len = meta_dim              # 57

    # 1) 시계열 피처 SHAP: reshape→평균(axis=0) → (1,62)
    shap_seq_vals  = shap_vals_for_avg[:seq_feat_len].reshape(seq_len, input_dim).mean(axis=0, keepdims=True)  # (1,62)
    input_seq_vals = avg_input[:, :seq_feat_len].reshape(seq_len, input_dim).mean(axis=0, keepdims=True)      # (1,62)

    plt.figure()
    shap.summary_plot(
        shap_seq_vals,
        input_seq_vals,
        feature_names=seq_feature_names,
        show=False
    )
    plt.title(f"Group{group_idx}: SHAP – Time-Series Features")
    fname_ts = f"shap_summary_time_series_group{group_idx}.png"
    plt.savefig(fname_ts, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {fname_ts}")

    # 2) 메타 피처 SHAP: (1,57)로 reshape
    shap_meta_vals  = shap_vals_for_avg[seq_feat_len:].reshape(1, meta_dim)   # (1,57)
    input_meta_vals = avg_input[:, seq_feat_len:].reshape(1, meta_dim)        # (1,57)

    plt.figure()
    shap.summary_plot(
        shap_meta_vals,
        input_meta_vals,
        feature_names=meta_feature_names,
        show=False
    )
    plt.title(f"Group{group_idx}: SHAP – Meta Features")
    fname_meta = f"shap_summary_meta_features_group{group_idx}.png"
    plt.savefig(fname_meta, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {fname_meta}")
