# ChurnTrack-XAI

# Customer Churn Prediction

본 프로젝트는 카드 고객의 월별 **시계열 거래 데이터**와 **정적 메타 정보**를 활용하여, 6개월째 고객 이탈 여부(`churn`)를 예측하는 딥러닝 모델을 구현합니다.  
모델의 **설명 가능성(XAI)** 강화를 위해 **SHAP**, **Integrated Gradients**, **Attention Heatmap** 분석 도구를 포함합니다.

---

## 구성 파일
```
정규화 적용 버전/
├── train_feature.py # 전체 학습 및 실행 스크립트
├── model.py # BiLSTM 기반 모델 정의
├── dataloader.py # 시계열 + 메타데이터 로딩 모듈
├── shap_3.py # SHAP 기반 설명 모듈 v0
├── inference_module.py # 저장된 모델을 활용한 예측 및 Attention 시각화 v0
├── processed_time_series.csv # 월별 고객 시계열 데이터
├── meta_merged.csv # 고객별 정적 정보 및 이탈 여부
├── requirements.txt # 프로젝트 실행 환경의 Python 패키지 목록
```
## 실행 방법
```bash
python train.py
```

## 데이터 설명

### 1. `processed_time_series.csv`
- 고객 ID별 월 단위 시계열 거래 특성 포함
- 총 5개월치 window 기반 학습 (예: 2018.07 ~ 2018.11 → 2018.12 이탈 예측)

### 2. `meta_merged.csv`
- 고객별 정적 속성 (성별, 연령, 카드 등급 등)
- `churn` 레이블 포함
- 클래스 불균형 해소를 위한 1:1 언더샘플링 처리

---

## 모델 개요
- BiLSTM: 시계열의 양방향 패턴 학습
- 1D CNN: 로컬 패턴 추출
- Attention: 시점별 중요도 가중합
- FC Layer: 메타 데이터 임베딩
- Classifier: Attention + Meta 정보 병합 → 최종 예측
```bash
(Time-series) → BiLSTM → CNN → Attention →────────┐
                                                  │concat
(Meta features) → FC ReLU →───────────────────────┘→ FC → churn probability
```
## 정규화 및 전처리 적용 사항
- bool 타입 → float 변환
- 결측값 보간: 시계열은 ffill + 0 대체, 메타는 중간값 대체
- 메타 임베딩에 대한 L2 정규화 항 추가 (meta_reg_lambda)

## XAI 기능 구성

### SHAP (`shap.py`)
- KernelExplainer 기반
- 시계열 / 메타 피처 중요도 분리 시각화
- 출력:  
  - `shap_summary_seq_all_l2.png`  
  - `shap_summary_meta_all_l2.png`

### Integrated Gradients (`ig.py`)
- Captum 기반
- False Positive, False Negative 외에도 **중간 확률(MID)**, **높은 확률(HIGH)** 그룹까지 분석
- 각 그룹별 **Top-10 중요 피처 시각화**
- 출력 파일:
  - `ig_meta_top10_FP_l1_v2.png`, `ig_seq_top10_FP_l1_v2.png`
  - `ig_meta_top10_FN_l1_v2.png`, `ig_seq_top10_FN_l1_v2.png`
  - `ig_meta_top10_MID_l1_v2.png`, `ig_seq_top10_MID_l1_v2.png`
  - `ig_meta_top10_HIGH_l1_v2.png`, `ig_seq_top10_HIGH_l1_v2.png`

### Attention Heatmap (`train.py`)
- Validation 전체 평균 attention 시각화
- 출력: `attention_heatmap_*.png`

## 실행 환경

아래 파일을 통해 본 프로젝트의 의존성 및 실행 환경을 확인할 수 있습니다:

- [`requirements.txt`](./정규화 적용 버전/requirements.txt)  
  → 주요 패키지:
  - `torch==2.5.1+cu121`
  - `torchvision==0.20.1+cu121`
  - `torchaudio==2.5.1+cu121`
  - `shap==0.47.2`
  - `captum==0.8.0`
  - `scikit-learn==1.6.1`
  - `matplotlib==3.9.4`
  - `pandas==2.2.3`
  - `numpy==1.26.4`

> 본 프로젝트는 **CUDA 12.1** 환경에서 실행되었으며, PyTorch는 `+cu121` 빌드를 사용합니다.
> `nvidia-smi` 기준 GPU 드라이버 버전은 **535.113 이상**을 권장합니다.


---
