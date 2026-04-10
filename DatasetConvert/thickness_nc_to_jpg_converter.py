#!/usr/bin/env python
import os
import re
from pathlib import Path

import numpy as np
import xarray as xr

# ================== 설정 ==================
# v206 NetCDF들이 있는 루트 디렉토리
V206_ROOT = Path("/home/yejun/projects/ipiu_2025/cs2smos_by_ym/v206")

# 출력 디렉토리 (원본/정규화)
OUT_ROOT = Path("/home/yejun/projects/ipiu_2025/cs2smos_v206_sit_npy")
OUT_RAW = OUT_ROOT / "raw"           # 원본 SIT (m)
OUT_NORM = OUT_ROOT / "normalized"   # 정규화된 SIT ([0,1])

OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT_NORM.mkdir(parents=True, exist_ok=True)

# SIT 변수 우선순위 (v206 구조 기준)
VAR_CAND = [
    "analysis_sea_ice_thickness",
    "weighted_mean_sea_ice_thickness",
    "background_sea_ice_thickness",
    "smos_sea_ice_thickness",
    "cryosat_sea_ice_thickness",
    "sea_ice_thickness",
    "sit",
]

# 정규화용 최대 두께 (m) – 0~5m로 clip한 뒤 /5
NORM_MAX_THICKNESS = 5.0
# ==========================================


def extract_date_from_filename(fname: str):
    """
    예: ..._20231015_20231021_r_v206_01_l4sit.nc → '20231021' 반환
    """
    m = re.search(r"_(\d{8})_(\d{8})_", fname)
    if not m:
        return None
    start, end = m.groups()
    return end  # 주로 end date를 대표 날짜로 사용


def find_sit_var(ds: xr.Dataset):
    """SIT 변수 이름을 VAR_CAND 순서대로 탐색"""
    for cand in VAR_CAND:
        if cand in ds.data_vars:
            return cand
    return None


def process_one_nc(nc_path: Path):
    fname = nc_path.name
    date_str = extract_date_from_filename(fname)

    if date_str is None:
        print(f"[WARN] 날짜를 파싱할 수 없음, 스킵: {fname}")
        return

    # 이미 해당 날짜의 파일이 있으면 스킵 (원하면 덮어쓰기 허용 가능)
    raw_out_path = OUT_RAW / f"{date_str}.npy"
    norm_out_path = OUT_NORM / f"{date_str}.npy"
    if raw_out_path.exists() and norm_out_path.exists():
        print(f"[SKIP] already exists: {date_str}")
        return

    print(f"[PROC] {fname} -> date={date_str}")

    ds = xr.open_dataset(nc_path)

    try:
        var_name = find_sit_var(ds)
        if var_name is None:
            print(f"  [WARN] SIT 변수를 찾지 못함, 스킵: {fname}")
            print(f"         vars = {list(ds.data_vars.keys())}")
            return

        da = ds[var_name]

        # time dimension 제거 (보통 time=1)
        if "time" in da.dims:
            da = da.isel(time=0)

        arr = da.values.astype(np.float32)

        # FillValue 처리
        fill = da.attrs.get("_FillValue", None)
        if fill is not None:
            arr = np.where(arr == fill, np.nan, arr)

        # 음수 두께는 물리적으로 무의미 → NaN
        arr[arr < 0] = np.nan

        if np.isnan(arr).all():
            print("  [WARN] 모든 값이 NaN, 스킵.")
            return

        # ---- 원본 저장 (m 단위) ----
        np.save(raw_out_path, arr)
        print(f"  -> saved raw SIT: {raw_out_path}")

        # ---- 정규화 ----
        # 0 ~ NORM_MAX_THICKNESS (예: 0~5m) 로 클리핑 후 [0,1] 스케일
        clipped = np.clip(arr, 0.0, NORM_MAX_THICKNESS)
        norm = clipped / NORM_MAX_THICKNESS

        np.save(norm_out_path, norm)
        print(f"  -> saved normalized SIT: {norm_out_path}")

    finally:
        ds.close()


def main():
    if not V206_ROOT.exists():
        print(f"[ERROR] V206_ROOT does not exist: {V206_ROOT}")
        return

    # v206 아래 모든 .nc 파일 재귀 탐색 (연/월/ 파일 구조 상관없이 싹 다)
    nc_files = sorted(V206_ROOT.rglob("*.nc"))
    print(f"[INFO] Found {len(nc_files)} .nc files under {V206_ROOT}")

    if not nc_files:
        print("[WARN] 처리할 .nc 파일이 없습니다.")
        return

    for i, nc_path in enumerate(nc_files, 1):
        print(f"\n[{i}/{len(nc_files)}] {nc_path}")
        process_one_nc(nc_path)

    print("\n[DONE] v206 SIT → date-based npy (raw + normalized) 변환 완료.")


if __name__ == "__main__":
    main()
