import re
import ftplib
from pathlib import Path

import numpy as np
import xarray as xr

# ================== 설정 ==================
FTP_HOST = "ftp.awi.de"

# 루트 경로 (버전 폴더들이 있는 곳)
CS2SMOS_ROOT = "/sea_ice/product/cryosat2_smos"

# 로컬 저장 위치
LOCAL_ROOT = Path("/home/yejun/projects/ipiu_2025/cs2smos_data")
RAW_DIR = LOCAL_ROOT / "raw"
PROC_DIR = LOCAL_ROOT / "processed"
# ==========================================


def connect_ftp():
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login()
    print(f"[INFO] Connected to {FTP_HOST} as anonymous")
    return ftp


def detect_latest_version_dir(ftp) -> str:
    """
    /sea_ice/product/cryosat2_smos 안에 있는 버전 폴더(v2xx)를 찾아서
    가장 최신(v숫자 최대) 폴더 이름을 리턴.
    예: 'v202', 'v204', ...
    """
    ftp.cwd(CS2SMOS_ROOT)
    names = ftp.nlst()
    # v뒤에 숫자가 오는 폴더만 골라냄
    vers = []
    for n in names:
        if re.match(r"^v\d+", n):
            vers.append(n)

    if not vers:
        raise RuntimeError(
            f"[ERROR] 버전 폴더(v2xx)를 찾지 못했습니다. 목록: {names}"
        )

    # 숫자 기준 정렬해서 가장 큰 것 선택
    vers_sorted = sorted(vers, key=lambda s: int(re.findall(r"\d+", s)[0]))
    latest = vers_sorted[-1]
    print(f"[INFO] Detected CS2SMOS version folder: {latest}")
    return latest


def find_nh_dir(ftp, version_dir: str) -> str:
    """
    해당 버전 폴더 안에서 북반구 디렉토리 경로를 찾는다.
    일반적으로:
      /sea_ice/product/cryosat2_smos/v2xx/nh
    혹은
      /sea_ice/product/cryosat2_smos/v2xx/nh/LATEST
    """
    base = f"{CS2SMOS_ROOT}/{version_dir}"
    ftp.cwd(base)
    sub = ftp.nlst()
    # nh 또는 NH 같은 이름 찾기
    nh_candidates = [s for s in sub if s.lower().startswith("nh")]
    if not nh_candidates:
        raise RuntimeError(f"[ERROR] {base} 안에서 nh 디렉토리를 찾지 못했습니다. 목록: {sub}")
    nh_dir = nh_candidates[0]
    nh_path = f"{base}/{nh_dir}"
    print(f"[INFO] Found NH folder: {nh_path}")

    # nh 폴더 안에 LATEST가 있으면 거기로
    ftp.cwd(nh_path)
    inside = ftp.nlst()
    if "LATEST" in inside:
        latest_path = f"{nh_path}/LATEST"
        print(f"[INFO] Using LATEST folder: {latest_path}")
        return latest_path
    else:
        # LATEST 없으면 nh 폴더 자체에서 파일을 다운로드
        print(f"[INFO] No LATEST folder, use: {nh_path}")
        return nh_path


def list_nc_files(ftp, target_dir: str):
    ftp.cwd(target_dir)
    names = ftp.nlst()
    # l4sit 이 들어간 netCDF만 선택
    nc_files = [
        n for n in names
        if n.lower().endswith(".nc") and "l4sit" in n.lower()
    ]
    print(f"[INFO] Found {len(nc_files)} l4sit .nc files in {target_dir}")
    return nc_files



def download_all(ftp, nc_files):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for fname in nc_files:
        local_path = RAW_DIR / fname
        if local_path.exists():
            print(f"[SKIP] {local_path} already exists")
            downloaded.append(local_path)
            continue

        print(f"[DL] {fname} -> {local_path}")
        with open(local_path, "wb") as f:
            ftp.retrbinary(f"RETR " + fname, f.write)

        downloaded.append(local_path)

    print(f"[INFO] Downloaded {len(downloaded)} files.")
    return downloaded


def find_sit_var(ds: xr.Dataset) -> xr.DataArray:
    cand = ["analysis_sea_ice_thickness", "sea_ice_thickness", "sit"]
    for name in cand:
        if name in ds.data_vars:
            return ds[name]
    raise ValueError(f"SIT 변수를 찾을 수 없음. vars = {list(ds.data_vars.keys())}")


def preprocess_file(nc_path: Path, lonlat_saved_flag: dict):
    print(f"[PROC] {nc_path}")
    ds = xr.open_dataset(nc_path)

    sit = find_sit_var(ds)

    if "time" in sit.dims:
        sit2d = sit.isel(time=0)
    else:
        sit2d = sit

    sit_np = sit2d.values.astype("float32")

    fill = sit2d.attrs.get("_FillValue", None)
    if fill is not None:
        sit_np = np.where(sit_np == fill, np.nan, sit_np)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_name = nc_path.stem + "_sit.npy"
    out_path = PROC_DIR / out_name
    np.save(out_path, sit_np)
    print(f"  -> saved SIT to {out_path}")

    # lon/lat 저장 (처음 한 번만)
    if not lonlat_saved_flag.get("done", False):
        lon = ds["lon"].values.astype("float32") if "lon" in ds else None
        lat = ds["lat"].values.astype("float32") if "lat" in ds else None

        if lon is not None and lat is not None:
            np.save(PROC_DIR / "lon.npy", lon)
            np.save(PROC_DIR / "lat.npy", lat)
            print("  -> saved lon.npy, lat.npy")
            lonlat_saved_flag["done"] = True
        else:
            print("  !! lon/lat 변수를 찾지 못해 lon/lat 저장 생략")


def preprocess_all(downloaded_paths):
    lonlat_saved_flag = {"done": False}
    for p in downloaded_paths:
        preprocess_file(p, lonlat_saved_flag)


def main():
    ftp = connect_ftp()
    try:
        version_dir = detect_latest_version_dir(ftp)
        nh_dir = find_nh_dir(ftp, version_dir)
        nc_files = list_nc_files(ftp, nh_dir)
        downloaded = download_all(ftp, nc_files)
    finally:
        ftp.quit()
        print("[INFO] FTP connection closed")

    preprocess_all(downloaded)
    print("[DONE] all preprocessing finished.")


if __name__ == "__main__":
    main()
