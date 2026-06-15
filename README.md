## Overview

본 프로젝트는 반도체 제조 공정에서 발생하는 웨이퍼(Wafer) 결함을 자동으로 분류하기 위한 딥러닝 기반 시스템이다.

WM-811K(LSWMD) 데이터셋을 활용하여 다양한 결함 패턴을 학습하였으며, 데이터 불균형 문제를 해결하기 위해 Conditional Wasserstein GAN with Gradient Penalty (CWGAN-GP)를 이용한 데이터 증강 기법을 적용하였다.

생성된 합성 데이터를 실제 데이터와 결합하여 CNN 기반 결함 분류 모델을 학습하고 성능을 평가하였다.

---

# Features

* WM-811K(LSWMD) 웨이퍼 데이터셋 사용
* Wafer Map One-Hot Encoding
* Data Augmentation 적용
* Conditional WGAN-GP 기반 결함 데이터 생성
* Synthetic Data + Real Data 결합 학습
* Deep CNN 기반 결함 분류
* Accuracy, Precision, Recall, F1-Score 평가
* Macro F1-Score 측정
* Inference Time 및 FPS 측정

---

# Dataset

## WM-811K (LSWMD)

WM-811K 데이터셋은 실제 반도체 제조 공정에서 수집된 웨이퍼 맵 데이터셋으로 다양한 결함 패턴을 포함한다.

### Classes

| Class     | Description           |
| --------- | --------------------- |
| none      | Normal                |
| Center    | Center Defect         |
| Donut     | Donut Defect          |
| Edge-Ring | Edge Ring Defect      |
| Edge-Loc  | Edge Localized Defect |
| Loc       | Local Defect          |
| Random    | Random Defect         |
| Scratch   | Scratch Defect        |
| Near-full | Near Full Defect      |

총 9개의 클래스를 대상으로 분류를 수행하였다.

---

# Project Structure

```text
LSWMD.pkl
    │
    ▼
dataset.py
(WaferDataset)
    │
    ├─────────────────────────────┐
    │                             │
    ▼                             ▼
train_wgan.py                train_v2.py
(WGAN Training)             (CNN Training)
    │                             ▲
    ▼                             │
cwgan_gp.py                      │
Generator / Critic               │
    │                             │
    ▼                             │
wgan_generator_epoch_200.pth     │
    │                             │
    ▼                             │
generate_data.py                 │
(Synthetic Data Generation)      │
    │                             │
    ▼                             │
synthetic_wafer_data.pt ──────────┘
    │
    ▼
eval.py
(Model Evaluation)
```

---

# Workflow

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
CWGAN-GP
(Data Generation)
   │
   ▼
Synthetic Dataset
   │
   ▼
Real + Synthetic Dataset
   │
   ▼
Deep CNN
(Classification)
   │
   ▼
Evaluation
```

---

# Preprocessing

## Image Resize

모든 웨이퍼 맵 이미지를 64×64 크기로 통일하였다.

```text
Variable Size
      ↓
64 × 64
```

## One-Hot Encoding

웨이퍼 상태 정보를 3채널 One-Hot 형태로 변환하였다.

```text
0 → Background
1 → Wafer
2 → Defect
```

## Data Augmentation

훈련 데이터에 대해 다음 증강 기법을 적용하였다.

* Random Horizontal Flip
* Random Vertical Flip
* Random Rotation

---

# Evaluation Metrics

본 연구에서는 다음 지표를 사용하여 성능을 평가하였다.

* Accuracy
* Precision
* Recall
* F1-Score
* Macro F1-Score
* Inference Time
* FPS

---

# Experimental Result

## Class-wise F1 Score

| Class     | F1 Score |
| --------- | -------- |
| None      | 0.98     |
| Center    | 0.63     |
| Donut     | 0.72     |
| Edge-Ring | 0.76     |
| Edge-Loc  | 0.64     |
| Loc       | 0.71     |
| Random    | 0.65     |
| Scratch   | 0.74     |
| Near-full | 0.77     |

### Macro F1 Score

**0.7333**

---

# Requirements

```bash
torch
torchvision
numpy
pandas
opencv-python
scikit-learn
matplotlib
seaborn
```


# Run

## 1. Train CWGAN-GP

```bash
python train_wgan.py
```

## 2. Generate Synthetic Data

```bash
python generate_data.py
```

## 3. Train CNN

```bash
python train_v2.py
```

## 4. Evaluate Model

```bash
python eval.py
```

---
