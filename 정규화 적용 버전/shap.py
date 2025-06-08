import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from model import BiLSTM_CNN_Attention
import os

# ──────────────────────────────────────
# 1. 폰트 설정 (한글 대응)
font_path = "fonts/NanumGothicCoding-2.0/나눔고딕코딩.ttf"
if not os.path.exists(font_path):
    font_path = "/home/danu/deep/fonts/NanumGothicCoding-2.0/나눔고딕코딩.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

# ──────────────────────────────────────
# 2. 데이터 로드
data = np.load("shap_input_test_201807_201812_20250608_002829.npz")
x_seq = torch.from_numpy(data["x_seq"]).float()
x_meta = torch.from_numpy(data["x_meta"]).float()
y = torch.from_numpy(data["y"]).float().squeeze()
seq_feature_names = data["seq_feature_names"].tolist()
meta_feature_names = data["meta_feature_names"].tolist()

input_dim = x_seq.shape[2]
seq_len = x_seq.shape[1]
meta_dim = x_meta.shape[1]

# ──────────────────────────────────────
# 3. 모델 로드
model = BiLSTM_CNN_Attention(
    input_dim=input_dim,
    meta_dim=meta_dim,
    hidden_dim=64,
    cnn_out_channels=64,
    kernel_size=3
)
model.load_state_dict(torch.load("best_churn_model_201807_201812_20250608_002829.pth", map_location="cpu"))
model.eval()

# ──────────────────────────────────────
# 4. SHAP 예측 함수 정의
def model_fn(input_array):
    batch_size = input_array.shape[0]
    arr = torch.from_numpy(input_array).float()
    x_seq_in = arr[:, :seq_len * input_dim].reshape(batch_size, seq_len, input_dim)
    x_meta_in = arr[:, seq_len * input_dim:]
    with torch.no_grad():
        output, _, _ = model(x_seq_in, x_meta_in)
        probs = torch.sigmoid(output.view(-1))
    return probs.numpy()

# ──────────────────────────────────────
# 5. SHAP 계산 대상 입력 구성
model_input = torch.cat([x_seq.reshape(len(x_seq), -1), x_meta], dim=1)
model_input_np = model_input.numpy()

# 6. Background 설정
bg_n = min(100, model_input_np.shape[0])
background = model_input_np[:bg_n]

# 7. SHAP 계산
explainer = shap.KernelExplainer(model_fn, background)
shap_vals_raw = explainer.shap_values(model_input_np, nsamples=100)
shap_values = shap_vals_raw[0] if isinstance(shap_vals_raw, list) else shap_vals_raw

if shap_values.ndim == 1:
    shap_values = shap_values.reshape(1, -1)
if shap_values.shape[0] != model_input_np.shape[0]:
    shap_values = shap_values.T  # (N, feature_dim)

# ──────────────────────────────────────
# 8. 시계열 / 메타 SHAP 분리 및 시각화
seq_feat_len = seq_len * input_dim
shap_seq_vals = shap_values[:, :seq_feat_len].reshape(-1, seq_len, input_dim).mean(axis=1)  # (N, input_dim)
shap_meta_vals = shap_values[:, seq_feat_len:]  # (N, meta_dim)

x_seq_mean = x_seq.mean(dim=1).numpy()  # (N, input_dim)
x_meta_np = x_meta.numpy()

# 시계열 Summary Plot
shap.summary_plot(
    shap_seq_vals,
    x_seq_mean,
    feature_names=seq_feature_names,
    show=False
)
plt.title("SHAP – 전체 시계열 피처")
plt.savefig("shap_summary_seq_all_l2.png", bbox_inches="tight", dpi=300)
plt.close()
print("Saved: shap_summary_seq_all.png")

# 메타 Summary Plot
shap.summary_plot(
    shap_meta_vals,
    x_meta_np,
    feature_names=meta_feature_names,
    show=False
)
plt.title("SHAP – 전체 메타 피처")
plt.savefig("shap_summary_meta_all_l2.png", bbox_inches="tight", dpi=300)
plt.close()
print("Saved: shap_summary_meta_all.png")
