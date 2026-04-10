import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ================== 설정 ==================
INPUT_DIR = "/home/yejun/projects/ipiu_2025/ice_type_dataset/"  # ice_edge nc들이 있는 상위 폴더
FILE_PATTERN = "**/*.nc"   # 하위 폴더까지 전부

OUTPUT_IMG_DIR = "./edge_color_jpg"
OUTPUT_REG_NPY_DIR = "./edge_regularized_npy"  # [-1,1] 정규화 npy
# ================== 설정 끝 ==================

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_REG_NPY_DIR, exist_ok=True)

def find_edge_var(ds):
    """Dataset 안에서 ice_edge 변수만 찾기."""
    if "ice_edge" in ds.data_vars:
        return "ice_edge"
    # ice_edge 없는 파일(ice_type 등)은 그냥 건너뜀
    return None

# 0=흰색, 1=보라, 2=초록, 3=노랑
purple = (0.4, 0.0, 0.6)
green  = (0.1, 0.7, 0.2)
yellow = (1.0, 0.9, 0.0)
white  = (1.0, 1.0, 1.0)
color_cmap = ListedColormap([white, purple, green, yellow])

nc_files = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN), recursive=True))
print(f"Found {len(nc_files)} nc files.")

for path in nc_files:
    ds = xr.open_dataset(path)

    var_name = find_edge_var(ds)
    if var_name is None:
        print(f"[SKIP] {path} (no ice_edge variable)")
        ds.close()
        continue

    print(f"Processing: {path}")

    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=0)

    # arr: 원래 ice_edge 값 (보통 1,2,3 + NaN)
    arr = np.squeeze(da.values.astype(np.float32))  # (yc, xc)

    # NaN / land 마스크
    valid = np.isfinite(arr)

    # ---- 클래스 라벨링: 0/1/2/3 (시각화용) ----
    labels = np.zeros_like(arr, dtype=np.int32)  # 기본 0 = no data (NaN/육지)

    v = arr[valid]

    lab = np.zeros_like(v, dtype=np.int32)
    lab[(v >= 0.5) & (v < 1.5)] = 1   # open-water
    lab[(v >= 1.5) & (v < 2.5)] = 2   # open ice
    lab[(v >= 2.5) & (v < 3.5)] = 3   # closed ice

    labels[valid] = lab

    # ---- [-1, 1]로 정규화된 배열 만들기 ----
    # 1 -> -1, 2 -> 0, 3 -> 1  (대칭이라 디퓨전에 쓰기 좋음)
    reg = np.full_like(arr, np.nan, dtype=np.float32)  # 육지는 NaN 그대로
    # valid 위치에서만 정규화
    reg_valid = (lab.astype(np.float32) - 2.0)  # 1->-1, 2->0, 3->1
    reg[valid] = reg_valid

    # ---- 파일명: YYYYMMDD ----
    filename = os.path.basename(path).replace(".nc", "")
    parts = filename.split("_")
    if parts[-1].isdigit():
        date_str = parts[-1][:8]  # YYYYMMDD
    else:
        date_str = filename

    img_path = os.path.join(OUTPUT_IMG_DIR, f"{date_str}.jpg")
    reg_npy_path = os.path.join(OUTPUT_REG_NPY_DIR, f"{date_str}.npy")

    # ---- 색깔 이미지 저장 (보라/초록/노랑) ----
    plt.figure(figsize=(6, 6))
    plt.imshow(labels, cmap=color_cmap, vmin=0, vmax=3)
    plt.axis("off")
    plt.savefig(img_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    # ---- 정규화 npy 저장 ----
    # 디퓨전 학습 시 float32 [-1,0,1] + NaN
    np.save(reg_npy_path, reg)

    print(f"  -> saved {img_path} / {reg_npy_path}")

    ds.close()

print("Done.")
