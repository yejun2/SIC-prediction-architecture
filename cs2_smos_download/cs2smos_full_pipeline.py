#!/usr/bin/env python
import ftplib
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xarray as xr

# ================== 설정 ==================
FTP_HOST = "ftp.awi.de"
CS2SMOS_ROOT = "/sea_ice/product/cryosat2_smos"

# 이 버전들 중, 실제 서버에 있는 것만 사용
TARGET_VERSIONS = ["v202", "v203", "v204", "v205", "v206", "v300"]

# 로컬 저장 위치 (원하면 여기만 바꾸면 됨)
LOCAL_ROOT = Path("/home/yejun/projects/ipiu_2025/cs2smos_data")
RAW_ROOT = LOCAL_ROOT / "raw"        # .nc 저장
PROC_ROOT = LOCAL_ROOT / "processed" # sit.npy, lon.npy, lat.npy 등

# 병렬 다운로드/전처리 스레드 수 (네트워크 상황 봐서 조절)
MAX_WORKERS_DOWNLOAD = 8
MAX_WORKERS_PROCESS  = 8
# ==========================================


# --------------------- FTP 유틸 --------------------- #
def connect_ftp():
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login()
    print(f"[INFO] Connected to {FTP_HOST} as anonymous")
    return ftp


def list_versions_on_server(ftp):
    """서버에 실제로 존재하는 버전 폴더 목록 리턴"""
    ftp.cwd(CS2SMOS_ROOT)
    names = ftp.nlst()
    versions = [n for n in names if re.match(r"^v\d+", n)]
    print("[INFO] Versions on server:", versions)
    return versions


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

    nh_candidates = [s for s in sub if s.lower().startswith("nh")]
    if not nh_candidates:
        raise RuntimeError(f"[ERROR] {base} 안에서 nh 디렉토리를 찾지 못했습니다. 목록: {sub}")
    nh_dir = nh_candidates[0]
    nh_path = f"{base}/{nh_dir}"
    print(f"[INFO] [{version_dir}] Found NH folder: {nh_path}")

    # nh 폴더 안에 LATEST가 있으면 거기로
    ftp.cwd(nh_path)
    inside = ftp.nlst()
    if "LATEST" in inside:
        latest_path = f"{nh_path}/LATEST"
        print(f"[INFO] [{version_dir}] Using LATEST folder: {latest_path}")
        return latest_path
    else:
        print(f"[INFO] [{version_dir}] No LATEST folder, use: {nh_path}")
        return nh_path


def list_sit_nc_files(ftp, target_dir: str):
    """
    target_dir 안의 l4sit .nc 파일 목록 리턴
    예: W_XX-ESA,SMOS_CS2_S3A_S3B,NH_12P5KM_EASE2_20231015_20231021_o_v300_01_l4sit.nc
    """
    ftp.cwd(target_dir)
    names = ftp.nlst()
    nc_files = [
        n for n in names
        if n.lower().endswith(".nc") and "l4sit" in n.lower()
    ]
    print(f"[INFO] Found {len(nc_files)} l4sit .nc files in {target_dir}")
    return nc_files


# ----------------- 다운로드 (병렬) ----------------- #
def download_one(remote_path: str, local_path: Path):
    """
    remote_path: 절대 경로 (예: /sea_ice/...)
    """
    if local_path.exists():
        print(f"[SKIP] already exists: {local_path}")
        return

    try:
        ftp = connect_ftp()
        dir_path, fname = remote_path.rsplit("/", 1)
        ftp.cwd(dir_path)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[DL] {remote_path} -> {local_path}")

        with open(local_path, "wb") as f:
            ftp.retrbinary("RETR " + fname, f.write)

        ftp.quit()
    except Exception as e:
        print(f"[ERROR] download failed for {remote_path}: {e}")


def download_all_sit_files():
    """
    1) 서버에서 실제 있는 버전 확인
    2) 그 중 TARGET_VERSIONS 와 겹치는 것만 사용
    3) 각 버전의 NH(/LATEST) 에서 l4sit .nc 리스트 수집
    4) 병렬 다운로드
    """
    ftp = connect_ftp()
    try:
        versions_on_server = set(list_versions_on_server(ftp))
        versions_to_use = [v for v in TARGET_VERSIONS if v in versions_on_server]

        if not versions_to_use:
            print("[ERROR] 서버에 TARGET_VERSIONS 중 존재하는 버전이 없습니다.")
            return []

        remote_local_pairs = []

        for v in versions_to_use:
            try:
                nh_dir = find_nh_dir(ftp, v)
                sit_files = list_sit_nc_files(ftp, nh_dir)
                for fname in sit_files:
                    remote_path = f"{nh_dir}/{fname}"
                    local_path = RAW_ROOT / v / fname
                    remote_local_pairs.append((remote_path, local_path))
            except Exception as e:
                print(f"[WARN] 버전 {v} 처리 중 에러 발생, 스킵: {e}")

    finally:
        ftp.quit()
        print("[INFO] FTP connection closed (listing phase)")

    # 병렬 다운로드
    print(f"[INFO] Total files to download: {len(remote_local_pairs)}")
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DOWNLOAD) as ex:
        futures = [
            ex.submit(download_one, r, lp)
            for (r, lp) in remote_local_pairs
        ]
        for fut in as_completed(futures):
            _ = fut.result()

    # 로컬 경로 리스트 리턴
    downloaded_paths = [lp for _, lp in remote_local_pairs]
    return downloaded_paths


# --------------- 전처리: SIT 추출 --------------- #
def find_sit_var(ds: xr.Dataset) -> xr.DataArray:
    """
    CS2SMOS 파일 안에서 sea-ice thickness 변수 찾기.
    버전에 따라 이름이 약간 다를 수 있음.
    """
    cand = [
        "analysis_sea_ice_thickness",
        "sea_ice_thickness",
        "sit",
    ]
    for name in cand:
        if name in ds.data_vars:
            return ds[name]
    raise ValueError(f"SIT 변수를 찾을 수 없음. vars = {list(ds.data_vars.keys())}")


def parse_dates_from_fname(fname: str):
    """
    파일 이름에서 YYYYMMDD_YYYYMMDD 패턴을 찾아 start, end 리턴
    예: ..._20231015_20231021_...
    """
    m = re.search(r"(\d{8})_(\d{8})", fname)
    if m:
        return m.group(1), m.group(2)
    else:
        return None, None


def preprocess_one(nc_path: Path, lonlat_flag_path: Path, meta_lines: list):
    """
    NetCDF -> SIT numpy 저장
    lon/lat은 한 번만 저장
    meta_lines 리스트에 메타데이터 문자열 append
    """
    try:
        print(f"[PROC] {nc_path}")
        ds = xr.open_dataset(nc_path)

        sit = find_sit_var(ds)

        if "time" in sit.dims:
            sit2d = sit.isel(time=0)
        else:
            sit2d = sit

        sit_np = sit2d.values.astype("float32")  # (Y, X)

        fill = sit2d.attrs.get("_FillValue", None)
        if fill is not None:
            sit_np = np.where(sit_np == fill, np.nan, sit_np)

        # 저장 경로 (버전별 하위 폴더)
        rel = nc_path.relative_to(RAW_ROOT)
        version_dir = rel.parts[0]  # v202, v203, ...
        out_dir = PROC_ROOT / version_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        out_name = nc_path.stem + "_sit.npy"
        out_path = out_dir / out_name
        np.save(out_path, sit_np)
        print(f"  -> saved SIT to {out_path}")

        # lon/lat 저장 (한 번만)
        if not lonlat_flag_path.exists():
            lon = None
            lat = None
            for cand_lon in ["lon", "longitude", "x"]:
                if cand_lon in ds:
                    lon = ds[cand_lon].values.astype("float32")
                    break
            for cand_lat in ["lat", "latitude", "y"]:
                if cand_lat in ds:
                    lat = ds[cand_lat].values.astype("float32")
                    break

            if lon is not None and lat is not None:
                np.save(PROC_ROOT / "lon.npy", lon)
                np.save(PROC_ROOT / "lat.npy", lat)
                lonlat_flag_path.write_text("done")
                print("  -> saved lon.npy, lat.npy")
            else:
                print("  !! lon/lat 변수를 찾지 못해 lon/lat 저장 생략")

        # 메타데이터: 파일 이름에서 날짜 추출
        start_str, end_str = parse_dates_from_fname(nc_path.name)
        meta_lines.append(
            f"{nc_path.name},{version_dir},{start_str},{end_str}\n"
        )

    except Exception as e:
        print(f"[ERROR] preprocess failed for {nc_path}: {e}")


def preprocess_all(downloaded_paths):
    """
    다운로드된 .nc 들을 전부 돌면서 SIT만 추출해 .npy로 저장.
    """
    PROC_ROOT.mkdir(parents=True, exist_ok=True)
    lonlat_flag_path = PROC_ROOT / ".lonlat_saved"

    # 실제 존재하는 파일들만 대상으로
    nc_paths = [p for p in downloaded_paths if p.exists()]
    if not nc_paths:
        print("[WARN] 전처리할 .nc 파일이 없습니다.")
        return

    meta_lines = []
    # 메타데이터를 concurrency-safe 하게 모으려면, 간단히 after loop에서 정리하는 방식 사용.
    # 여기서는 각 스레드에서 append 하도록 하고, race는 조금 허용 (중복 가능성이 매우 낮음).
    # 혹시 엄격하게 하고 싶으면 queue 사용 or 파일 락 구현 가능.

    def worker(path):
        preprocess_one(path, lonlat_flag_path, meta_lines)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PROCESS) as ex:
        futures = [ex.submit(worker, p) for p in nc_paths]
        for fut in as_completed(futures):
            _ = fut.result()

    # 메타데이터 CSV 저장
    meta_path = PROC_ROOT / "cs2smos_sit_meta.csv"
    with meta_path.open("w") as f:
        f.write("filename,version,start_date,end_date\n")
        for line in meta_lines:
            f.write(line)
    print(f"[INFO] Saved metadata to {meta_path}")


# ------------------------- main ------------------------- #
def main():
    # 1) SIT L4 파일 전체 병렬 다운로드
    downloaded_paths = download_all_sit_files()

    # 2) 전처리 (sit.npy, lon/lat, meta.csv)
    preprocess_all(downloaded_paths)

    print("[DONE] CS2SMOS full pipeline finished.")


if __name__ == "__main__":
    main()
