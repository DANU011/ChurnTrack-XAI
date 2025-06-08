import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from captum.attr import IntegratedGradients
from model_seq_only import BiLSTM_CNN_Attention

# ───────────────────────────────────────────────────────
# 1. 한글 폰트 설정
font_path = "fonts/NanumGothicCoding-2.0/나눔고딕코딩.ttf"
if not os.path.exists(font_path):
    font_path = "/home/danu/deep/fonts/NanumGothicCoding-2.0/나눔고딕코딩.ttf"
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# ───────────────────────────────────────────────────────
# 2. 데이터 및 모델 로드
shap_data = np.load("shap_input_test_201807_201812_20250607_081509.npz", allow_pickle=True)
x_seq = shap_data["x_seq"]  # [N, T, seq_dim]
y = shap_data["y"]
seq_feature_names = shap_data["seq_feature_names"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTM_CNN_Attention(
    input_dim=x_seq.shape[2],
    hidden_dim=64,
    cnn_out_channels=64,
    kernel_size=3
).to(device)
model.load_state_dict(torch.load("best_churn_model_201807_201812_20250607_081509.pth", map_location=device))
model.train()

# ───────────────────────────────────────────────────────
# 3. Custom Forward 함수 (x_seq만)
def custom_forward(x_seq):
    model.train()
    logits, _ = model(x_seq)
    return logits

ig = IntegratedGradients(custom_forward)

# ───────────────────────────────────────────────────────
# 4. 잘못 분류된 샘플 인덱스 추출
with torch.no_grad():
    model.eval()
    inputs_seq_tensor = torch.tensor(x_seq, dtype=torch.float32).to(device)
    logits, _ = model(inputs_seq_tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    y_pred = (probs >= 0.5).astype(int)

fp_indices = np.where((y == 0) & (y_pred == 1))[0]  # False Positive
fn_indices = np.where((y == 1) & (y_pred == 0))[0]  # False Negative

# ───────────────────────────────────────────────────────
# 5. IG 실행 및 시각화 함수 (시계열 only)
def run_ig_and_plot(index, label):
    input_seq = torch.tensor(x_seq[index:index + 1], dtype=torch.float32).to(device)
    baseline_seq = torch.zeros_like(input_seq).to(device)

    attributions_seq = ig.attribute(
        inputs=input_seq,
        baselines=baseline_seq,
        target=0,
        return_convergence_delta=False
    )

    # ───── 시계열 피처 중요도 Top 10 시각화 ─────
    attr_seq = attributions_seq.squeeze().cpu().detach().numpy()  # [T, seq_dim]
    avg_attr_per_feature = np.mean(attr_seq, axis=0)  # [seq_dim]

    top_indices = np.argsort(np.abs(avg_attr_per_feature))[::-1][:10]
    top_attr = avg_attr_per_feature[top_indices]
    top_names = [seq_feature_names[i] for i in top_indices]

    plt.figure(figsize=(8, 4))
    plt.bar(top_names, top_attr, color="skyblue")
    plt.xticks(rotation=45)
    plt.title(f"IG - {label} 시계열 피처 중요도 (상위 10개)")
    plt.tight_layout()
    plt.savefig(f"ig_seq_top10_{label}_seq.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ig_seq_top10_{label}_seq.png")

# ───────────────────────────────────────────────────────
# 6. 실행 (False Positive / False Negative)
if len(fp_indices) > 0:
    run_ig_and_plot(fp_indices[0], "FP")
if len(fn_indices) > 0:
    run_ig_and_plot(fn_indices[0], "FN")
