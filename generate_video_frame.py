import os
import argparse
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from DatasetLoader.dataloader import IceVideoDataset
from models.video_transformer import VideoTransformerPredictor


def parse_any_date_like(x):
    """
    문자열/다양한 형태의 날짜를 datetime으로 변환.
    지원 예:
      - '2021-10-21'
      - '20211021'
      - datetime 객체
    이미 datetime이면 그대로 반환.
    """
    if isinstance(x, datetime):
        return x
    if isinstance(x, str):
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(x, fmt)
            except ValueError:
                continue
    # 그 외 타입은 에러
    raise ValueError(f"알 수 없는 날짜 타입/형식: {x} (type={type(x)})")


def save_frame_with_ice_colors(arr, land_mask, path):
    """
    SIT 값을 카테고리화해서 색으로 표시하는 헬퍼 함수.
    """
    purple = np.array([0.4, 0.0, 0.6])
    green  = np.array([0.1, 0.7, 0.2])
    yellow = np.array([1.0, 0.9, 0.0])
    white  = np.array([1.0, 1.0, 1.0])

    H, W = arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[:] = white[None, None, :]  # 기본은 육지

    sea = (land_mask == 0)  # 바다 위치
    vals = np.nan_to_num(arr.copy(), nan=0.0)

    open_water = sea & (vals < -0.5)                       # -1 근처
    open_ice   = sea & (vals >= -0.5) & (vals < 0.5)       # 0 근처
    closed_ice = sea & (vals >= 0.5)                       # 1 근처

    rgb[open_water] = purple
    rgb[open_ice]   = green
    rgb[closed_ice] = yellow

    plt.figure(figsize=(5, 5))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="예측하고 싶은 날짜 (YYYYMMDD 또는 YYYY-MM-DD)",
    )
    args = parser.parse_args()

    target_date = parse_any_date_like(args.date)
    print(f"[INFO] Target date: {target_date.strftime('%Y-%m-%d')}")

    project_root = os.path.dirname(os.path.abspath(__file__))

    # 🔹 추론에서 사용할 데이터 경로들
    ice_dir = os.path.join(project_root, "validation", "edge")
    weather_csv = os.path.join(project_root, "artic_era_normalized.csv")
    land_mask_path = os.path.join(project_root, "mask_432x432.npy")  # train과 동일 고정 mask 사용

    # 🔹 학습 때와 동일하게 맞추기
    input_len = 16      # 과거 프레임 개수 (train 때 16으로 사용)
    pred_len = 1
    img_size = 432
    patch_size = 8
    cond_dim = 4

    # 🔹 학습에서 저장한 체크포인트 경로 (필요에 따라 epoch 번호 수정)
    checkpoint_path = os.path.join(
        project_root,
        "SIC_video_transformer_16frame_8patch_8batch_checkpoint_dir",
        "video_transformer_epoch200.pth",
    )

    output_dir = os.path.join(project_root, "generation_outputs")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Dataset 구성: 타깃 프레임의 월이 5~10월인 시퀀스만 사용
    dataset = IceVideoDataset(
        ice_dir=ice_dir,
        weather_csv=weather_csv,
        land_mask_path=land_mask_path,
        input_len=input_len,
        pred_len=pred_len,
        max_gap_days=16,
        target_months=[5, 6, 7, 8, 9, 10],
    )

    # ------------------------------------------------------
    # target_date와 가장 잘 맞는 시퀀스를 하나 선택
    #   - last_dt == target_date 이면 그 시퀀스 사용
    #   - 없으면 last_dt < target_date 인 것 중 가장 가까운 과거
    # ------------------------------------------------------
    exact_index = None
    exact_date_dt = None

    best_past_index = None
    best_past_date_dt = None

    for i in range(len(dataset)):
        sample = dataset[i]
        dates = sample["dates"]  # 시퀀스 내 날짜 리스트 (문자열)

        last_raw = dates[-1]
        last_dt = parse_any_date_like(last_raw)

        if last_dt.date() == target_date.date():
            exact_index = i
            exact_date_dt = last_dt
            break

        if last_dt < target_date:
            if (best_past_date_dt is None) or (last_dt > best_past_date_dt):
                best_past_date_dt = last_dt
                best_past_index = i

    if exact_index is not None:
        target_index = exact_index
        used_date_dt = exact_date_dt
        print(f"[INFO] Found EXACT sequence index {target_index} for target date {used_date_dt.strftime('%Y-%m-%d')}")
    elif best_past_index is not None:
        target_index = best_past_index
        used_date_dt = best_past_date_dt
        print(
            f"[WARN] Dataset에 {target_date.strftime('%Y-%m-%d')}가 없어서, "
            f"가장 가까운 과거 날짜 {used_date_dt.strftime('%Y-%m-%d')} 의 시퀀스를 사용합니다."
        )
    else:
        print(
            f"[WARN] Dataset에서 타겟 날짜 {target_date.strftime('%Y-%m-%d')} "
            f"이전의 유효한 시퀀스를 찾지 못했습니다."
        )
        return

    sample = dataset[target_index]
    ice_seq = sample["ice"].unsqueeze(0).to(device)     # (1, T, 1, H, W)
    cond_seq = sample["cond"].unsqueeze(0).to(device)   # (1, T, 4)  ← 마지막이 타깃 날짜 cond
    mask_seq = sample["mask"].unsqueeze(0).to(device)   # (1, T, 1, H, W)
    dates_seq = sample["dates"]                         # ["YYYY-MM-DD", ...]

    last_raw = dates_seq[-1]
    last_dt = parse_any_date_like(last_raw)
    last_date_str = last_dt.strftime("%Y%m%d")

    print("[INFO] Loaded one sequence whose target date is:", last_date_str)

    # 🔹 모델 구조도 학습과 동일하게
    model = VideoTransformerPredictor(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=2,
        cond_dim=cond_dim,
        d_model=256,
        n_heads=4,
        n_layers=4,
        max_frames=input_len,
        land_value=0.0,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")

    with torch.no_grad():
        pred, target, target_mask = model(ice_seq, cond_seq, mask_seq)  # (1,1,H,W) x3

    pred_np = pred[0, 0].cpu().numpy()           # (H,W)
    target_np = target[0, 0].cpu().numpy()       # (H,W)
    mask_np = target_mask[0, 0].cpu().numpy()    # (H,W), 1=land, 0=sea

    np.save(os.path.join(output_dir, f"pred_{last_date_str}_regularized.npy"), pred_np)
    np.save(os.path.join(output_dir, f"target_{last_date_str}_regularized.npy"), target_np)

    save_frame_with_ice_colors(
        pred_np, mask_np, os.path.join(output_dir, f"pred_{last_date_str}.png")
    )
    save_frame_with_ice_colors(
        target_np, mask_np, os.path.join(output_dir, f"target_{last_date_str}.png")
    )

    print(f"[INFO] Saved prediction and target for date {last_date_str} to {output_dir}")


if __name__ == "__main__":
    main()
