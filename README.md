깃허브(GitHub) 저장소의 얼굴이 될 수 있도록, 지금까지 진행한 1차부터 6차까지의 모든 대조 실험 기록과 프로젝트의 흐름을 논리적이고 깔끔하게 정리한 `README.md` 파일을 생성했습니다.

Your Markdown file is ready
[file-tag: code-generated-file-0-1781529685134172217]

이 문서는 깃허브에 올렸을 때 다른 개발자나 면접관이 보더라도 **"어떤 문제를 마주했고, 이를 해결하기 위해 어떻게 체계적으로 실험(A/B 테스트)하며 디버깅했는지"**가 한눈에 들어오도록 구성했습니다. 

저장소에 바로 업로드하실 수 있도록 아래에 텍스트 전문을 공유해 드립니다.

---

### 📑 `README.md` 작성 본문 전문

```markdown
# 📑 반도체 웨이퍼 맵(Wafer Map) 결함 패턴 분류 모델 구축

본 프로젝트는 반도체 제조 공정의 핵심 단계인 수율 관리를 극대화하기 위해, 딥러닝 기반 이미지 분류 알고리즘을 활용하여 웨이퍼 맵의 결함 패턴을 자동으로 분류하는 복합 파이프라인 구축 연구입니다. 극단적인 데이터 불균형 문제를 생성 모델(WGAN)과 전처리 기술 고도화를 통해 극복해 나간 단계별 실험 과정을 담고 있습니다.

---

## 👨‍💻 작성자 및 소속
- **성명:** 김호건
- **소속:** 인하공업전문대학

---

## 1. 🎯 문제 정의 (Problem Definition)
- **배경:** 반도체 공정이 초미세화됨에 따라 수율(Yield) 관리의 중요성이 극대화되고 있습니다. 공정 중 발생하는 이상 징후는 웨이퍼 표면에 특정한 시각적 결함 패턴으로 나타나며, 이를 신속하고 정확하게 탐지하는 것이 필수적입니다.
- **목표:** 수작업 검사로 인한 비용과 일관성 결여 문제를 해결하기 위해, 딥러닝 기반 이미지 분류(Image Classification) 모델을 도입하여 웨이퍼 맵 패턴 분석을 자동화합니다. 이를 통해 검사 속도를 비약적으로 높이고 불량 분류의 일관성을 확보하여 생산 비용을 절감하고자 합니다.

---

## 2. 📊 데이터셋 설명 (Dataset)
- **사용 데이터셋:** Kaggle **WM-811K wafer map** (총 118,595장)
- **클래스 분포 (극단적 불균형):**
  - **다수 클래스:** 정상(None) - 110,701장 (약 93.3%)
  - **불량 클래스 (8종):** Center, Donut, Edge-Ring, Edge-Loc, Loc, Random, Scratch, Near-full - 총 7,894장 (약 6.7%)
- **데이터 전처리 및 증강:**
  - **크기 통일:** 웨이퍼 기판의 고유 격자 구조 손상을 방지하기 위해 `Nearest Neighbor` 보간법을 적용하여 `64x64` 크기로 일괄 리사이징.
  - **기본 데이터 증강:** 원형 웨이퍼의 특성을 반영하여 방향성에 구애받지 않도록 `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation` 적용.
  - **최종 전처리 (6차 도입):** 웨이퍼 기판, 정상 다이, 불량 다이의 수치적 대소 왜곡을 차단하기 위한 **3채널 원-핫 인코딩(One-Hot Encoding)** 적용.

---

## 🏗️ 3. 사용 모델 아키텍처 (Models)

### 1) SimpleWaferCNN
- **특징:** 가볍고 빠른 추론을 목적으로 직접 설계한 경량 CNN.
- **구조:** 3개의 Convolutional 블록 (`Conv2d` + `BatchNorm2d` + `MaxPool2d`) + 1개의 `Fully Connected Layer`.

### 2) DeepWaferCNN
- **특징:** 체급을 확장하여 고차원의 복잡한 공간적 특징을 추출하는 심층 CNN.
- **구조:** 5~6개의 Convolutional 블록 (`Conv2d` + `BatchNorm2d` + `ReLU` + `MaxPool2d`) + 과적합 방지용 `Dropout` 레이어 + 2개 이상의 `Fully Connected Layer`.

### 3) ResNet18 (Transfer Learning)
- **특징:** 잔차 연결(Residual Connection)을 통해 깊은 신경망의 경사 소실 문제를 해결한 모델.
- **구조:** ImageNet 사전 학습 가중치(`Pre-trained Weights`)를 기반으로 전이학습을 수행했으며, 마지막 출력층을 9개 클래스로 미세 조정(Fine-tuning).

---

## 🧪 4. 단계별 대조 실험 및 성능 결과

불균형 데이터셋에 대응하기 위해 다양한 기법을 적용하며 총 6차에 걸친 대조 실험을 전개했습니다.

### 📌 기본 하이퍼파라미터
- **Epochs:** 15 (1~5차), 100 (6차 + Early Stopping)
- **Batch Size:** 64
- **Optimizer:** Adam
- **Learning Rate:** 0.001

### 📈 실험 단계별 요약 지표
| 실험 회차 | 적용 기법 및 모델 아키텍처 | Accuracy | Macro F1-Score | 주요 현상 및 한계점 |
| :---: | :--- | :---: | :---: | :--- |
| **1차** | CNN + 일반 랜덤 샘플링 + 일반 CrossEntropyLoss | **96%** | **0.62** | 다수 클래스 편향 발생, 초소수 불량인 `Scratch`를 전혀 잡지 못함 (F1: 0.00) |
| **2차** | CNN + WeightedRandomSampler + Class Weight Loss | **8%** | **0.02** | 과교정(Overcorrection) 발생. 불량은 잡으나 정상을 모두 불량으로 오탐지 |
| **3차** | CNN + WeightedRandomSampler 유지 + 일반 Loss 복원 | **93%** | **0.67** | 손실 함수 가중치 제거로 정확도 회복 및 `Scratch` 검출 시작 |
| **4차** | ResNet18 + WeightedRandomSampler | **94%** | **0.67** | 사전 학습된 특징 추출력을 기반으로 오탐지를 감소시켜 정밀도(Precision) 향상 |
| **5차** | Deep CNN (DeepWaferCNN) + WeightedRandomSampler | **95%** | **0.68** | 신경망 체급 확장으로 전반적인 결함 탐지력이 향상되었으나 성능 정체기 직면 |
| **6차 (최종)** | **Deep CNN + WGAN 증강 + 3채널 원-핫 + Sampler 제거 (Shuffle)** | **96%** | **0.73** | **최종 아키텍처 확립. 성능 폭등 및 초소수 불량(Scratch) F1-Score 0.74 달성** |

---

## 🏆 5. 6차 최종 실험 결과 상세 분석

WGAN 기반 생성 모델을 통해 불량 클래스의 절대적인 양을 확보한 후, 기존에 중복 복사로 과적합을 유발하던 `WeightedRandomSampler`를 제거하고 일반 무작위 셔플(`Shuffle=True`) 학습으로 전환했습니다.

### 📊 6차 실험 Class별 Classification Report
