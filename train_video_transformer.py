import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # ✅ tqdm

from DatasetLoader.dataloader import IceVideoDataset
from models.video_transformer import VideoTransformerPredictor


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    ice_dir = os.path.abspath(os.path.join(project_root, "cs2smos_v206_sit_npy", "normalized"))
    weather_csv = os.path.abspath(os.path.join(project_root, "artic_era_normalized.csv"))
    land_mask_path = os.path.abspath(os.path.join(project_root, "mask_432x432.npy"))

    # 🔹 과거 16프레임 + 미래 1프레임
    input_len = 16         # 과거 프레임 개수
    pred_len = 1           # 예측 프레임 개수
    batch_size = 1         # A6000에서 키우기
    num_epochs = 500
    lr = 5e-5              # 🔽 살짝 낮춘 learning rate

    img_size = 432
    patch_size = 8         # 432 / 8 = 54 패치
    cond_dim = 4           # [sst_mean, t2m_mean, d2m_mean, tp_mean]

    save_dir = os.path.join(
        project_root, "SIC_video_transformer_16frame_8patch_8batch_checkpoint_dir"
    )
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print((f"[INFO] Using device: {device}"))

    # 1. Dataset / DataLoader
    #    - target_months=[5..10] → 타깃(마지막 프레임 날짜)이 5~10월인 시퀀스만 사용
    dataset = IceVideoDataset(
        ice_dir=ice_dir,
        weather_csv=weather_csv,
        land_mask_path=land_mask_path,
        input_len=input_len,
        pred_len=pred_len,
        max_gap_days=16,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[INFO] Dataset size: {len(dataset)} sequences")

    # 2. 모델 생성
    model = VideoTransformerPredictor(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=2,     # [ice_filled, sea_mask]
        cond_dim=cond_dim, # cond: (B, T, 4)  (마지막 T-1이 타깃 cond)
        d_model=256,       # A6000에서 여유되면 512
        n_heads=4,
        n_layers=4,
        max_frames=input_len,  # history frame 수
        land_value=0.0,
    ).to(device)

    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # (선택) Mixed Precision (AMP) – GPU일 때만 켜기
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    global_step = 0

    # 4. 학습 루프
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0     # ✅ 유효한 배치 수만 카운트

        # tqdm으로 DataLoader 감싸기
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)

        for i, batch in enumerate(pbar):
            ice = batch["ice"].to(device)    # (B, T, 1, H, W), T = 16+1
            cond = batch["cond"].to(device)  # (B, T, 4)  ← 타깃 날짜 cond 포함
            mask = batch["mask"].to(device)  # (B, T, 1, H, W)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # 모델 forward
                # VideoTransformerPredictor 내부에서 history/target을 나눠 쓴다고 가정
                pred, target, target_mask = model(ice, cond, mask)  # (B,1,H,W) x3
                loss = model.masked_mse_loss(pred, target, target_mask)

            # 🔍 NaN / Inf 체크: 문제가 있으면 이 배치 스킵
            if not torch.isfinite(loss):
                print(
                    f"[WARN] Non-finite loss detected at epoch {epoch}, "
                    f"step {i}: {loss.detach().item()}"
                )
                print("       → Skipping this batch (no update)")
                optimizer.zero_grad(set_to_none=True)
                continue

            # backward
            scaler.scale(loss).backward()

            # ✅ 기울기 폭주 방지: unscale 후 gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1
            global_step += 1

            # tqdm progress bar에 현재 loss 표시 (유효한 배치만 평균)
            avg_loss = running_loss / max(num_batches, 1)
            pbar.set_postfix({"loss": f"{avg_loss:.6f}"})

        # 5. epoch 끝날 때마다 체크포인트 저장
        ckpt_path = os.path.join(save_dir, f"video_transformer_epoch{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(f"[INFO] Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
