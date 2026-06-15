import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from cnn_models import DeepWaferCNN


def load_and_preprocess_npz(npz_path, target_size=(64, 64)):
    print(f"📦 Kaggle .npz 데이터셋 로드 중: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    print(f"🔍 포함된 데이터 키 목록: {list(data.keys())}")

    wafer_maps = data["arr_0"]
    failure_types = data["arr_1"]

    print("wafer_maps shape:", wafer_maps.shape)
    print("failure_types shape:", failure_types.shape)
    print("failure_types 앞 10개:")
    print(failure_types[:10])

    target_names = [
        "none",
        "Center",
        "Donut",
        "Edge-Ring",
        "Edge-Loc",
        "Loc",
        "Random",
        "Scratch",
        "Near-full"
    ]

    external_to_model = {
        0: ("Center", 1),
        1: ("Donut", 2),
        2: ("Edge-Loc", 4),
        3: ("Edge-Ring", 3),
        4: ("Loc", 5),
        5: ("Near-full", 8),
        6: ("Scratch", 7),
        7: ("Random", 6),
    }


    external_label_names = [
        "Center",
        "Donut",
        "Edge-Ring",
        "Edge-Loc",
        "Loc",
        "Random",
        "Scratch",
        "Near-full"
    ]

    filtered_imgs = []
    filtered_labels = []

    single_count = 0
    multi_or_other_count = 0

    external_class_count = {
        name: 0 for name, _ in external_to_model.values()
    }

    for img, f_type in zip(wafer_maps, failure_types):
        f_type = np.array(f_type)
        label_sum = int(np.sum(f_type))

        if label_sum == 1:
            external_idx = int(np.argmax(f_type))

            if external_idx not in external_to_model:
                multi_or_other_count += 1
                continue

            external_name, model_label = external_to_model[external_idx]

            filtered_imgs.append(img)
            filtered_labels.append(model_label)

            single_count += 1
            external_class_count[external_name] += 1

        else:
            multi_or_other_count += 1

    print("\n🔎 단일 결함 클래스 분포:")
    for name, count in external_class_count.items():
        print(f"  {name}: {count}")

    print(f"\n🎯 단일 결함 데이터: {single_count}장")
    print(f"🚫 혼합/정상/기타 제외 데이터: {multi_or_other_count}장")

    if len(filtered_imgs) == 0:
        raise ValueError(
            "단일 결함 데이터가 0장입니다. "
            "failure_types가 one-hot 라벨인지 확인하세요."
        )

    print("\n🔄 이미지를 모델 입력 규격(1, 64, 64)에 맞게 변환 중...")

    tensor_imgs = []

    for img in filtered_imgs:
        img = np.array(img)

        if img.size == 0:
            continue

        # (H, W) → (1, 1, H, W)
        t_img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 52x52 → 64x64
        t_img_resized = F.interpolate(
            t_img,
            size=target_size,
            mode="nearest"
        )

        # (1, 1, 64, 64) → (1, 64, 64)
        tensor_imgs.append(t_img_resized.squeeze(0))

    if len(tensor_imgs) == 0:
        raise ValueError("리사이즈 후 남은 이미지가 0장입니다.")

    X_test = torch.stack(tensor_imgs)
    y_test = torch.tensor(filtered_labels, dtype=torch.long)

    # 기존 dataset.py와 동일하게 0,1,2 → 0,0.5,1 스케일로 변환
    if X_test.max() > 1.0:
        X_test = X_test / 2.0

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("X_test min/max:", X_test.min().item(), X_test.max().item())

    return X_test, y_test, target_names


def run_external_test():
    NPZ_PATH = "Wafer_Map_Datasets.npz"
    MODEL_PATH = "best_wafer_model.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"평가 장비: {DEVICE}")

    try:
        X_test, y_test, target_names = load_and_preprocess_npz(NPZ_PATH)
    except KeyError as e:
        print(f"❌ 에러 발생: npz key 이름을 확인하세요. {e}")
        return
    except ValueError as e:
        print(f"❌ 에러 발생: {e}")
        return

    test_dataset = TensorDataset(X_test, y_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    print(f"\n🧠 DeepWaferCNN 모델을 불러옵니다: {MODEL_PATH}")

    model = DeepWaferCNN(num_classes=9).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    all_preds = []
    all_targets = []

    print("\n🚀 외부 데이터셋 평가를 시작합니다...")

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    print("\n📊 [외부 데이터셋 검증 결과]")
    print("=" * 60)

    print(
        classification_report(
            all_targets,
            all_preds,
            labels=list(range(9)),
            target_names=target_names,
            zero_division=0
        )
    )

    print("=" * 60)

    cm = confusion_matrix(
        all_targets,
        all_preds,
        labels=list(range(9))
    )

    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    run_external_test()