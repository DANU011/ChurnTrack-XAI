# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# import os
# from captum.attr import IntegratedGradients
# from model import BiLSTM_CNN_Attention
# """
# 생성파일
#
# ig_meta_top10_FP.png
# ig_seq_top10_FP.png
# ig_meta_top10_FN.png
# ig_seq_top10_FN.png
# """
# # ───────────────────────────────────────────────────────
# # 1. 한글 폰트 설정 (Matplotlib)
# font_path = "fonts/NanumGothicCoding-2.0/나눔고딕코딩.ttf"
# if not os.path.exists(font_path):
#     font_path = "/home/danu/deep/fonts/NanumGothicCoding-2.0/나눔고딕코딩.ttf"
# fm.fontManager.addfont(font_path)
# font_name = fm.FontProperties(fname=font_path).get_name()
# plt.rcParams['font.family'] = font_name
# plt.rcParams['axes.unicode_minus'] = False
#
# # ───────────────────────────────────────────────────────
# # 2. 데이터 및 모델 로드
# shap_data = np.load("shap_input_201807_201812_20250608_065227.npz", allow_pickle=True)
# x_seq = shap_data["x_seq"]  # [N, T, seq_dim]
# x_meta = shap_data["x_meta"]  # [N, meta_dim]
# y = shap_data["y"]
# seq_feature_names = shap_data["seq_feature_names"]
# meta_feature_names = shap_data["meta_feature_names"]
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = BiLSTM_CNN_Attention(
#     input_dim=x_seq.shape[2],
#     meta_dim=x_meta.shape[1],
#     hidden_dim=64,
#     cnn_out_channels=64,
#     kernel_size=3
# ).to(device)
# model.load_state_dict(torch.load("best_churn_model_201807_201812_20250608_061524.pth", map_location=device))
# model.train()
#
# # ───────────────────────────────────────────────────────
# # 3. Custom Forward 함수 (logits만 반환)
# def custom_forward(x_seq, x_meta):
#     model.train()
#     logits, _, _ = model(x_seq, x_meta)
#     return logits
#
# ig = IntegratedGradients(custom_forward)
#
# # ───────────────────────────────────────────────────────
# # 4. 잘못 분류된 샘플 인덱스 추출
# with torch.no_grad():
#     model.eval()
#     inputs_seq_tensor = torch.tensor(x_seq, dtype=torch.float32).to(device)
#     inputs_meta_tensor = torch.tensor(x_meta, dtype=torch.float32).to(device)
#     logits, _, _ = model(inputs_seq_tensor, inputs_meta_tensor)
#     probs = torch.sigmoid(logits).squeeze().cpu().numpy()
#     y_pred = (probs >= 0.5).astype(int)
#
# fp_indices = np.where((y == 0) & (y_pred == 1))[0]  # False Positive
# fn_indices = np.where((y == 1) & (y_pred == 0))[0]  # False Negative
#
# # ───────────────────────────────────────────────────────
# # 5. IG 실행 및 시각화 함수
# def run_ig_and_plot(index, label):
#     input_seq = torch.tensor(x_seq[index:index + 1], dtype=torch.float32).to(device)
#     input_meta = torch.tensor(x_meta[index:index + 1], dtype=torch.float32).to(device)
#     baseline_seq = torch.zeros_like(input_seq).to(device)
#     baseline_meta = torch.zeros_like(input_meta).to(device)
#
#     attributions_seq, attributions_meta = ig.attribute(
#         inputs=(input_seq, input_meta),
#         baselines=(baseline_seq, baseline_meta),
#         target=0,
#         return_convergence_delta=False
#     )
#
#     # ───── 메타데이터 중요도 Top 10 시각화 ─────
#     attr_meta = attributions_meta.squeeze().cpu().detach().numpy()
#     sorted_idx = np.argsort(np.abs(attr_meta))[::-1][:10]
#     plt.figure(figsize=(8, 4))
#     plt.bar(np.array(meta_feature_names)[sorted_idx], attr_meta[sorted_idx], color="orange")
#     plt.xticks(rotation=45)
#     plt.title(f"IG - {label} 메타 중요도 (상위 10개)")
#     plt.tight_layout()
#     plt.savefig(f"ig_meta_top10_{label}_phase424.png", dpi=300, bbox_inches='tight')
#     plt.close()
#
#     # ───── 시계열 피처 중요도 Top 10 시각화 ─────
#     attr_seq = attributions_seq.squeeze().cpu().detach().numpy()  # [T, seq_dim]
#     avg_attr_per_feature = np.mean(attr_seq, axis=0)  # [seq_dim]
#
#     top_indices = np.argsort(np.abs(avg_attr_per_feature))[::-1][:10]
#     top_attr = avg_attr_per_feature[top_indices]
#     top_names = [seq_feature_names[i] for i in top_indices]
#
#     plt.figure(figsize=(8, 4))
#     plt.bar(top_names, top_attr, color="skyblue")
#     plt.xticks(rotation=45)
#     plt.title(f"IG - {label} 시계열 피처 중요도 (상위 10개)")
#     plt.tight_layout()
#     plt.savefig(f"ig_seq_top10_{label}_phase424.png", dpi=300, bbox_inches='tight')
#     plt.close()
#
# # ───────────────────────────────────────────────────────
# # 6. 실행 (False Positive / False Negative 각각)
# if len(fp_indices) > 0:
#     run_ig_and_plot(fp_indices[0], "FP")
# if len(fn_indices) > 0:
#     run_ig_and_plot(fn_indices[0], "FN")
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from captum.attr import IntegratedGradients
from model import BiLSTM_CNN_Attention

# ───────────────────────────────────────────────────────
# 1. 한글 폰트 설정 (Matplotlib)
font_path = "fonts/NanumGothicCoding-2.0/나눔고딕코딩.ttf"
if not os.path.exists(font_path):
    font_path = "/home/danu/deep/fonts/NanumGothicCoding-2.0/나눔고딕코딩.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

# ───────────────────────────────────────────────────────
# 2. 데이터 및 모델 로드
shap_data = np.load("shap_input_test_201807_201812_20250608_012201.npz", allow_pickle=True)
x_seq = shap_data["x_seq"]
x_meta = shap_data["x_meta"]
y = shap_data["y"]
seq_feature_names = shap_data["seq_feature_names"]
meta_feature_names = shap_data["meta_feature_names"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTM_CNN_Attention(
    input_dim=x_seq.shape[2],
    meta_dim=x_meta.shape[1],
    hidden_dim=64,
    cnn_out_channels=64,
    kernel_size=3
).to(device)
model.load_state_dict(torch.load("best_churn_model_201807_201812_20250608_012201.pth", map_location=device))
model.eval()

# ───────────────────────────────────────────────────────
# 3. Custom Forward 함수 (logits만 반환)
def custom_forward(x_seq, x_meta):
    model.train()
    logits, _, _ = model(x_seq, x_meta)
    return logits

ig = IntegratedGradients(custom_forward)

# ───────────────────────────────────────────────────────
# 4. 예측 및 분류 결과
with torch.no_grad():
    inputs_seq_tensor = torch.tensor(x_seq, dtype=torch.float32).to(device)
    inputs_meta_tensor = torch.tensor(x_meta, dtype=torch.float32).to(device)
    logits, _, _ = model(inputs_seq_tensor, inputs_meta_tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    y_pred = (probs >= 0.5).astype(int)

# 분류 기준
fp_indices = np.where((y == 0) & (y_pred == 1))[0]
fn_indices = np.where((y == 1) & (y_pred == 0))[0]
mid_prob_indices = np.where((probs >= 0.5) & (probs < 0.8))[0]
high_prob_indices = np.where(probs >= 0.8)[0]

# ───────────────────────────────────────────────────────
# 5. IG 실행 및 시각화 함수
def run_ig_and_plot(index, label):
    input_seq = torch.tensor(x_seq[index:index+1], dtype=torch.float32).to(device)
    input_meta = torch.tensor(x_meta[index:index+1], dtype=torch.float32).to(device)
    baseline_seq = torch.zeros_like(input_seq).to(device)
    baseline_meta = torch.zeros_like(input_meta).to(device)

    attributions_seq, attributions_meta = ig.attribute(
        inputs=(input_seq, input_meta),
        baselines=(baseline_seq, baseline_meta),
        target=0,
        return_convergence_delta=False
    )

    # ───── 메타 피처 중요도 Top 10 ─────
    attr_meta = attributions_meta.squeeze().cpu().detach().numpy()
    sorted_idx = np.argsort(np.abs(attr_meta))[::-1][:10]
    plt.figure(figsize=(8, 4))
    plt.bar(np.array(meta_feature_names)[sorted_idx], attr_meta[sorted_idx], color="orange")
    plt.xticks(rotation=45)
    plt.title(f"IG - {label} 메타 중요도 (상위 10개)")
    plt.tight_layout()
    plt.savefig(f"ig_meta_top10_{label}_l1_v2.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ───── 시계열 피처 중요도 Top 10 ─────
    attr_seq = attributions_seq.squeeze().cpu().detach().numpy()
    avg_attr_per_feature = np.mean(attr_seq, axis=0)
    top_indices = np.argsort(np.abs(avg_attr_per_feature))[::-1][:10]
    top_attr = avg_attr_per_feature[top_indices]
    top_names = [seq_feature_names[i] for i in top_indices]

    plt.figure(figsize=(8, 4))
    plt.bar(top_names, top_attr, color="skyblue")
    plt.xticks(rotation=45)
    plt.title(f"IG - {label} 시계열 중요도 (상위 10개)")
    plt.tight_layout()
    plt.savefig(f"ig_seq_top10_{label}_l1_v2.png", dpi=300, bbox_inches='tight')
    plt.close()

# ───────────────────────────────────────────────────────
# 6. 분석 실행
for index_list, tag in zip(
    [fp_indices, fn_indices, mid_prob_indices, high_prob_indices],
    ["FP", "FN", "MID", "HIGH"]
):
    if len(index_list) > 0:
        run_ig_and_plot(index_list[0], tag)
