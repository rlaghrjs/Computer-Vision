import torch
import numpy as np
import matplotlib.pyplot as plt

PT_PATH = "synthetic_wafer_data.pt"

label_names = [
    "none", "Center", "Donut", "Edge-Ring",
    "Edge-Loc", "Loc", "Random", "Scratch", "Near-full"
]

data = torch.load(PT_PATH, map_location="cpu")

print("저장 타입:", type(data))

if isinstance(data, dict):
    print("키 목록:", data.keys())
    images = data["images"]
    labels = data["labels"]
else:
    images, labels = data

print("\n[기본 정보]")
print("images type:", type(images))
print("labels type:", type(labels))
print("images shape:", images.shape)
print("labels shape:", labels.shape)
print("images dtype:", images.dtype)
print("labels dtype:", labels.dtype)

print("\n[값 범위]")
print("images min:", images.min().item())
print("images max:", images.max().item())
print("images mean:", images.mean().item())
print("images std:", images.std().item())

print("\n[클래스 분포]")
labels_np = labels.numpy()
counts = np.bincount(labels_np, minlength=len(label_names))

for i, count in enumerate(counts):
    print(f"{i} ({label_names[i]}): {count}장")

print("\n[원-핫/확률맵 확인]")
sample = images[0]
channel_sum = sample.sum(dim=0)

print("샘플 1개 shape:", sample.shape)
print("픽셀별 채널합 min:", channel_sum.min().item())
print("픽셀별 채널합 max:", channel_sum.max().item())
print("픽셀별 채널합 mean:", channel_sum.mean().item())

# 샘플 이미지 저장
num_show = min(16, len(images))
plt.figure(figsize=(8, 8))

for i in range(num_show):
    img = images[i]
    label = labels[i].item()

    # [3,64,64] 확률맵을 [64,64] 클래스맵으로 변환
    img_class = torch.argmax(img, dim=0).numpy()

    plt.subplot(4, 4, i + 1)
    plt.imshow(img_class, cmap="gray")
    plt.title(f"{label_names[label]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("synthetic_samples.png", dpi=200)
print("\n샘플 이미지 저장 완료: synthetic_samples.png")