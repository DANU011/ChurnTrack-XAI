# ChurnTrack-XAI

# Customer Churn Prediction

이 프로젝트는 카드 고객의 시계열 거래 데이터와 정적 메타 데이터를 기반으로, 6개월째 이탈 여부(`churn`)를 예측하는 LSTM 기반 모델입니다.

---

## 구성 파일
```
.
├── main.py # 전체 학습 및 실행 스크립트
├── model.py # BiLSTM 기반 모델 정의
├── dataloader.py # 시계열 + 메타데이터 로딩 모듈
├── shap_explainer.py # SHAP 기반 설명 모듈 v0
├── inference_module.py # 저장된 모델을 활용한 예측 및 Attention 시각화 v0
├── processed_time_series.csv # 월별 고객 시계열 데이터
├── meta_merged.csv # 고객별 정적 정보 및 이탈 여부
```
## 실행 방법
```bash
python train.py
```

## 데이터 설명

### 1. `processed_time_series.csv`
- 고객 ID별 월 단위 시계열 거래 특성 포함

### 2. `meta_merged.csv`
- 고객별 정적 속성 (성별, 연령, 카드 등급 등) 및 `churn` 레이블 포함

---

## 모델 개요
- LSTM으로 시계열 정보 학습
- CNN으로 로컬 패턴 추출
- Attention으로 시점별 중요도 가중치 적용
- 정적 메타 정보와 병합 후 최종 이탈 여부 예측

---
