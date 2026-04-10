import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ================== 설정 ==================
INPUT_DIR = "/home/yejun/projects/ipiu_2025/ice_type_dataset/"  # ice_edge nc들이 있는 상위 폴더
FILE_PATTERN = "**/*.nc"   # 하위 폴더까지 전부 탐색

OUTPUT_IMG_DIR      = "/home/yejun/projects/ipiu_2025/edge_color_jpg"     # 색깔 jpg
OUTPUT_REG_NPY_DIR  = "/home/yejun/projects/ipiu_2025/edge_regularized_npy"  # [-1,0,1]+NaN
OUTPUT_MASK_NPY_DIR = "/home/yejun/projects/ipiu_2025/edge_maskmap_npy"      # land=1, sea=0
# ================== 설정 끝 ==================

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_REG_NPY_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_NPY_DIR, exist_ok=True)


def find_edge_var(ds):
    """Dataset 안에서 ice_edge 변수만 찾기."""
    if "ice_edge" in ds.data_vars:
        return "ice_edge"
    # ice_edge 없는 파일(ice_type 등)은 그냥 건너뜀
    return None


# 0=흰색(육지/NaN), 1=보라(open water), 2=초록(open ice), 3=노랑(closed ice)
purple = (0.4, 0.0, 0.6)
green  = (0.1, 0.7, 0.2)
yellow = (1.0, 0.9, 0.0)
white  = (1.0, 1.0, 1.0)
color_cmap = ListedColormap([white, purple, green, yellow])  # 0,1,2,3 순서

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

    # time 차원이 있으면 첫 번째 타임스텝만 사용
    if "time" in da.dims:
        da = da.isel(time=0)

    # arr: 원래 ice_edge 값 (보통 1,2,3 + NaN)
    arr = np.squeeze(da.values.astype(np.float32))  # (yc, xc)
    # valid: 바다/해빙 (유효한 숫자)
    valid = np.isfinite(arr)

    # -----------------------------
    # 1) 시각화용 label 생성 (0/1/2/3)
    #    0 = NaN/육지, 1 = open-water, 2 = open ice, 3 = closed ice
    # -----------------------------
    labels = np.zeros_like(arr, dtype=np.int32)  # 기본 0 = 육지/NaN

    v = arr[valid]  # 유효한 값만
    lab_valid = np.zeros_like(v, dtype=np.int32)

    # ice_edge 값 구간에 따라 클래스 부여 (필요시 조건 수정 가능)
    lab_valid[(v >= 0.5) & (v < 1.5)] = 1   # open-water
    lab_valid[(v >= 1.5) & (v < 2.5)] = 2   # open ice
    lab_valid[(v >= 2.5) & (v < 3.5)] = 3   # closed ice

    labels[valid] = lab_valid  # NaN/육지는 0 유지

    # -----------------------------
    # 2) [-1, 0, 1]로 정규화된 배열 만들기
    #    1 -> -1, 2 -> 0, 3 -> 1 (육지는 NaN 그대로 유지)
    # -----------------------------
    reg = np.full_like(arr, np.nan, dtype=np.float32)  # 육지는 NaN 유지

    # arr 값이 1,2,3이라고 가정할 때, (v - 2)로 매핑:
    # 1 -> -1, 2 -> 0, 3 -> 1
    reg_valid = v.astype(np.float32) - 2.0
    reg[valid] = reg_valid

    # -----------------------------
    # 3) land mask 생성
    #    land=1, sea=0 (NaN 기준)
    # -----------------------------
    mask = np.isnan(arr).astype(np.uint8)  # 육지=1, 바다=0

    # -----------------------------
    # 4) 파일명: YYYYMMDD 추출
    # -----------------------------
    filename = os.path.basename(path).replace(".nc", "")
    parts = filename.split("_")

    # 예: something_ice_edge_20210102.nc → parts[-1] = "20210102"
    # 또는 단순히 20210102.nc → parts[-1] = "20210102"
    if parts[-1].isdigit():
        date_str = parts[-1][:8]  # YYYYMMDD
    else:
        # 혹시 모를 예외: 파일명 전체가 날짜일 때
        # ex) 20210102.nc
        date_str = filename[:8]

    # img_path      = os.path.join(OUTPUT_IMG_DIR,     f"{date_str}.jpg")
    # reg_npy_path  = os.path.join(OUTPUT_REG_NPY_DIR, f"{date_str}.npy")
    mask_npy_path = os.path.join(OUTPUT_MASK_NPY_DIR,f"{date_str}.npy")

    # -----------------------------
    # 5) 색깔 이미지 저장 (보라/초록/노랑/흰색)
    # -----------------------------
    
    # plt.figure(figsize=(6, 6))
    # plt.imshow(labels, cmap=color_cmap, vmin=0, vmax=3)
    # plt.axis("off")
    # plt.savefig(img_path, dpi=150, bbox_inches="tight", pad_inches=0)
    # plt.close()

    # -----------------------------
    # 6) 정규화 npy + mask npy 저장
    # -----------------------------
    #np.save(reg_npy_path, reg)    # float32, [-1,0,1] + NaN(육지)
    np.save(mask_npy_path, mask)  # uint8, 육지=1, 바다=0

    # print(f"  -> saved {img_path}")
    # print(f"  -> saved {reg_npy_path}")
    print(f"  -> saved {mask_npy_path}")

    ds.close()

print("Done.")
