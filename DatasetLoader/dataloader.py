# DatasetLoader/dataloader.py

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def parse_date_from_filename(path: str) -> datetime:
    base = os.path.basename(path)        # e.g. '20210102.npy'
    date_str = base.replace(".npy", "")  # '20210102'
    return datetime.strptime(date_str, "%Y%m%d")


class IceVideoDataset(Dataset):
    """
    SIT(또는 SIC) npy + ERA5 cond + 고정 land mask 를 사용하는 비디오 시퀀스 Dataset.
    """

    def __init__(
        self,
        ice_dir: str,
        weather_csv: str,
        land_mask_path: str = None,  # 하나의 파일 경로 (없으면 나중에 0-mask 생성)
        input_len: int = 4,
        pred_len: int = 1,
        max_gap_days: int = 16,      # 날짜 점프 허용 최대 간격(일)
    ):
        super().__init__()

        self.ice_dir = ice_dir
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_len = input_len + pred_len
        self.max_gap_days = max_gap_days

        # 1) ice npy 파일 정렬
        ice_paths = sorted(glob.glob(os.path.join(ice_dir, "*.npy")))
        if len(ice_paths) == 0:
            raise ValueError(f"No npy files found in {ice_dir}")

        self.frames = []
        for p in ice_paths:
            d = parse_date_from_filename(p)  # datetime(YYYY,MM,DD)
            self.frames.append((d, p))       # (datetime, path)

        # 2) weather csv 로드 → 날짜별 daily mean
        df = pd.read_csv(weather_csv)

        if "time" not in df.columns:
            raise ValueError(
                f"'time' column not found in {weather_csv}. "
                f"columns={list(df.columns)}"
            )

        df["time"] = pd.to_datetime(df["time"])
        df["date"] = df["time"].dt.date

        self.weather_cols = ["sst_mean", "t2m_mean", "d2m_mean", "tp_mean"]
        for col in self.weather_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in weather csv. "
                    f"Available columns: {list(df.columns)}"
                )

        daily = df.groupby("date")[self.weather_cols].mean()
        daily.index = pd.to_datetime(daily.index)
        self.weather_df = daily

        # 3) land mask (고정 하나)
        if land_mask_path is not None:
            if not os.path.exists(land_mask_path):
                raise FileNotFoundError(f"Mask file not found: {land_mask_path}")

            mask_np = np.load(land_mask_path)  # (H,W) or (1,H,W)
            if mask_np.ndim == 2:
                mask_np = mask_np[None, ...]   # (1,H,W)
            self.static_mask = mask_np.astype(np.float32)  # (1, H, W)
        else:
            self.static_mask = None

        # 4) 유효 시퀀스 인덱스 (날짜 gap만 체크)
        self.valid_indices = []
        for i in range(len(self.frames)):
            end_idx = i + self.total_len - 1
            if end_idx >= len(self.frames):
                break

            ok = True
            for k in range(i, end_idx):
                d1 = self.frames[k][0]
                d2 = self.frames[k + 1][0]
                if (d2 - d1).days > self.max_gap_days:
                    ok = False
                    break

            if ok:
                self.valid_indices.append(i)

        if len(self.valid_indices) == 0:
            raise ValueError(
                "No valid sequences found. "
                "Check input_len/pred_len, data length, and max_gap_days."
            )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.total_len  # exclusive

        seq_frames = self.frames[start_idx:end_idx]
        dates = [f[0] for f in seq_frames]
        paths = [f[1] for f in seq_frames]
        T = len(dates)

        # 1) ice 시퀀스
        ice_list = []
        for p in paths:
            arr = np.load(p)
            if arr.ndim == 2:
                arr = arr[None, ...]
            ice_list.append(arr)
        ice_np = np.stack(ice_list, axis=0)   # (T, 1, H, W)
        ice_tensor = torch.from_numpy(ice_np).float()

        # 2) weather cond 시퀀스 (모든 T 프레임, 마지막이 타깃 cond)
        cond_list = []
        for d in dates:
            if d not in self.weather_df.index:
                raise ValueError(
                    f"Weather data missing for date {d.date()} "
                    f"(check csv date range and ice npy dates)."
                )
            cond_vals = (
                self.weather_df.loc[d, self.weather_cols]
                .values
                .astype(np.float32)
            )
            cond_list.append(cond_vals)
        cond_np = np.stack(cond_list, axis=0)  # (T, cond_dim)
        cond_tensor = torch.from_numpy(cond_np).float()

        # 3) land mask 시퀀스
        if self.static_mask is not None:
            mask_np = np.repeat(self.static_mask[None, ...], T, axis=0)
        else:
            _, _, H, W = ice_np.shape
            mask_np = np.zeros((T, 1, H, W), dtype=np.float32)
        mask_tensor = torch.from_numpy(mask_np).float()

        date_strs = [d.strftime("%Y-%m-%d") for d in dates]

        return {
            "ice": ice_tensor,    # (T, 1, H, W)
            "cond": cond_tensor,  # (T, 4)
            "mask": mask_tensor,  # (T, 1, H, W)
            "dates": date_strs,
        }
