# ChurnTrack-XAI

# Customer Churn Prediction

본 프로젝트는 카드 고객의 월별 **시계열 거래 데이터**와 **정적 메타 정보**를 활용하여, 6개월째 고객 이탈 여부(`churn`)를 예측하는 딥러닝 모델을 구현합니다.  
모델의 **설명 가능성(XAI)** 강화를 위해 **SHAP**, **Integrated Gradients**, **Attention Heatmap** 분석 도구를 포함합니다.

---

## 구성 파일
```
정규화 적용 버전/
├── train.py # 전체 학습 및 실행 스크립트
├── model.py # BiLSTM 기반 모델 정의
├── dataloader.py # 시계열 + 메타데이터 로딩 모듈
├── shap.py # SHAP 기반 설명 모듈
├── ig.py # Integrated Gradients 기반 피처 중요도 시각화 모듈
├── processed_time_series.csv # 월별 고객 시계열 데이터
├── meta_merged.csv # 고객별 정적 정보 및 이탈 여부
└── requirements.txt # 프로젝트 실행 환경의 Python 패키지 목록
```
## 실행 방법
```bash
python train.py
```
## 데이터 출처 및 피처 선정

본 프로젝트는 **AI허브**에서 제공하는  
[금융 합성 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?searchKeyword=%EA%B8%88%EC%9C%B5&aihubDataSe=data&dataSetSn=71792)을 기반으로 구축되었습니다.

해당 데이터셋은 카드사에서 수집한 월별 승인매출정보, 청구정보, 잔액정보, 회원정보 등을 포함하고 있습니다.

### 주요 원천 테이블 구성
- 카드 회원정보
- 카드 신용용정보
- 카드 청구정보
- 카드 잔액정보
- 카드 채널정보
- 카드 마케팅정보
- 카드 성과정보
- 개인 CB정보

### EDA 및 피처 선정 기준
EDA(탐색적 데이터 분석)를 통해 다음과 같은 전처리 및 피처 엔지니어링을 수행하였습니다:

- 이상값 제거 및 결측값 보완
- 주요 변수 기반 파생 피처 생성
- 시계열/메타 특성별 통계적 유의성 고려하여 피처 선별

### 최종 활용 피처 유형
| 피처 유형      | 설명 |
|---------------|------|
| 시계열 특성     | 거래금액, 이용건수, 잔액, 청구금액 등 월별 통계 기반 피처 |
| 메타 특성       | 성별, 연령, 카드 등급, 가입일, 채널 등 고객 고유 속성 |
| 이탈 레이블     | 기준월 기준 **6개월 째**의 이탈 여부 (`churn`) |

> 모델 학습은 총 5개월간 데이터를 기반으로 이루어지며,  
> **5개월치 시계열 입력 → 다음달(`6개월째`) 이탈 여부 예측** 구조로 구성됩니다.


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
(Time-series) → BiLSTM → CNN → Attention ─────┐
                                              ↓
                                     Concatenate → FC → churn probability
                                              ↑
           (Meta features) → FC (ReLU) ───────┘
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

## 추가 모델 조합 가이드
본 프로젝트는 다양한 구조의 모델 및 어텐션 메커니즘을 실험할 수 있도록 설계되어 있습니다. 정규화 적용 버전 외 모델 아키텍처 및 입력 구성을 변경할 때는 아래 기준에 따라 `train_*.py`, `model_*.py`, `attention_module_*.py`, `dataloader_*.py` 파일들을 조합하여 사용하십시오.
| 실험 유형                | 훈련 스크립트                | 모델 정의 파일                        | 어텐션 모듈                                     | 데이터로더 파일                 |
| -------------------- | ---------------------- | ------------------------------- | ------------------------------------------ | ------------------------ |
| 기본 모델                | `train_customer_id.py` | `model.py`                      | `attention_module.py`                      | `dataloader.py`          |
| Meta-aware Attention | `train_customer_id.py` | `model_meta_aware_attention.py` | `attention_module_meta_aware_attention.py` | `dataloader.py`          |
| Gating 구조 실험         | `train_customer_id.py` | `model_gating.py`               | `attention_module.py`                          | `dataloader.py`          |
| 시계열만 사용 (Seq-only)   | `train_seq_only.py`    | `model_seq_only.py`             | `attention_module.py`                         | `dataloader_seq_only.py` |
| Phase 학습 실험         | `train_phase.py`         | `model.py`      | `attention_module.py`     | `dataloader.py`              |

SHAP / IG 해석 도구는 모든 모델 실험 후 `.pth` 모델 가중치를 불러와 분석 가능하며,  
`shap_4.py`, `ig.py` 파일로 구성되어 있습니다.

실험 목적에 맞는 `train_*.py` 스크립트를 사용하고,  
내부에서 참조하는 모델과 데이터로더를 위 표에 맞게 조정하십시오.
> 예시:  
> Meta-aware Attention 실험을 하려면 `train_customer_id.py`에서  
> `model_meta_aware_attention.py`와 `attention_module_meta_aware_attention.py`를 불러오도록 수정합니다.

### 추가 모델 아키텍쳐
Meta-aware Attention 모델 구조
```bash
(Time-series) → BiLSTM → CNN ─────────────┐
                                          ↓
                      (Meta features) → FC (ReLU)
                                          ↓
                        Meta-aware Attention Layer
                                          ↓
                                     Concatenate → FC → churn probability
```

Gating 구조 모델
```bash
(Time-series) → BiLSTM → CNN ────────────────┐
                                             ↓
(Meta features) → FC (ReLU) → FC → sigmoid ──┘
                                             ↓
        Element-wise Gating (cnn_out ⊙ meta_gate) → Transpose → Attention → FC → churn probability

```

시계열 전용 모델 (Seq-only)
```bash
(Time-series) → BiLSTM → CNN → Attention → FC → churn probability
```

Phase 학습 실험 구조 (3단계: 시계열-only → 메타-only → 전체 fine-tuning)
```bash
Phase 1: 시계열 전용 학습
(Time-series) → BiLSTM → CNN → Attention → FC → churn probability

Phase 2: 메타 전용 학습
(Meta features) → FC (ReLU) → FC → churn probability

Phase 3: 통합 모델 fine-tuning
(Time-series) → BiLSTM → CNN → Attention ─────┐
                                              ↓
                                     Concatenate → FC → churn probability
                                              ↑
           (Meta features) → FC (ReLU) ───────┘
```

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
