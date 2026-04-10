import os
import glob

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ------------- 설정 -------------
INPUT_DIR = "/home/yejun/projects/ipiu_2025/ice_type_dataset/"
OUTPUT_DIR = "./dataset_icetype_jpg_clear"
FILE_PATTERN = "*.nc"
# --------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 실제 NetCDF 파일에서 나타날 가능성이 높은 변수명들
VAR_CANDIDATES = [
    "sea_ice_edge",
    "ice_edge",
    "Sea_ice_edge",
    "Sea_ice_edge_flag",
    "ice_edge_class",
    "ice_type"
]

# 변환할 파일 목록 찾기
nc_files = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN)))
print(f"Found {len(nc_files)} nc files.")

for path in nc_files:
    print(f"Processing: {path}")

    # 1) NetCDF 열기
    ds = xr.open_dataset(path)

    # 2) 변수 이름 찾기
    var_name = None
    for cand in VAR_CANDIDATES:
        if cand in ds.data_vars:
            var_name = cand
            break

    # 못 찾으면 경고 후 첫 변수 사용
    if var_name is None:
        var_name = list(ds.data_vars)[0]
        print(f"[WARN] No candidate variable found. Using first variable: {var_name}")

    da = ds[var_name]

    # 3) time 축 있으면 첫 번째 선택
    if "time" in da.dims:
        da = da.isel(time=0)

    img = da.values.astype(float)

    # 4) NaN → 특정 값으로 대체
    mask = ~np.isfinite(img)
    img[mask] = np.nan

    # Sea ice edge는 카테고리형 (1,2,3) → 스케일링 불필요
    vmin = 1
    vmax = 3

    # 5) 이미지 저장 (colorbar 없음)
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.axis("off")

    out_name = os.path.splitext(os.path.basename(path))[0] + ".jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    ds.close()

    print(f"  -> saved: {out_path}")

print("Done.")
