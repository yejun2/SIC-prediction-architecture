#!/usr/bin/env python
import ftplib
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================== 설정 ==================
FTP_HOST = "ftp.awi.de"
CS2SMOS_ROOT = "/sea_ice/product/cryosat2_smos"

# 이 버전들에 대해 다운로드 (필요하면 수정)
TARGET_VERSIONS = ["v204", "v205", "v206"]

# 로컬 저장 위치 (원하면 여기만 바꾸면 됨)
LOCAL_ROOT = Path("/home/yejun/projects/ipiu_2025/cs2smos_by_ym")

# 병렬 다운로드 스레드 수 (네트워크/서버 상황 봐서 4~16 정도 추천)
MAX_WORKERS_DOWNLOAD = 8
# ==========================================


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


def list_years(ftp, version_dir: str):
    """
    /.../v20x/nh 아래 연도 디렉토리 리스트 반환 (2010, 2011, ...)
    """
    base = f"{CS2SMOS_ROOT}/{version_dir}/nh"
    ftp.cwd(base)
    names = ftp.nlst()
    years = [n for n in names if n.isdigit() and len(n) == 4]
    print(f"[INFO] [{version_dir}] years under nh:", years)
    return base, years


def list_months(ftp, year_path: str):
    """
    /.../v20x/nh/YYYY 아래 월 디렉토리 리스트 반환 (01, 02, ..., 12)
    """
    ftp.cwd(year_path)
    names = ftp.nlst()
    months = [n for n in names if n.isdigit() and len(n) == 2]
    return months


def list_nc_files_in_month(ftp, month_path: str):
    """
    /.../v20x/nh/YYYY/MM 안의 .nc 파일 리스트 반환
    """
    ftp.cwd(month_path)
    names = ftp.nlst()
    nc_files = [n for n in names if n.lower().endswith(".nc")]
    return nc_files


def download_one(remote_path: str, local_path: Path):
    """
    remote_path: 절대 경로 (예: /sea_ice/...)
    local_path: 저장할 로컬 경로
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


def main():
    # 1) 버전/연/월 구조 탐색해서 전체 파일 리스트 만들기
    ftp = connect_ftp()
    remote_local_pairs = []

    try:
        versions_on_server = set(list_versions_on_server(ftp))
        versions_to_use = [v for v in TARGET_VERSIONS if v in versions_on_server]

        if not versions_to_use:
            print("[ERROR] 서버에 TARGET_VERSIONS 중 존재하는 버전이 없습니다.")
            return

        print("[INFO] Using versions:", versions_to_use)

        for v in versions_to_use:
            try:
                base, years = list_years(ftp, v)
            except Exception as e:
                print(f"[WARN] {v} nh 구조 탐색 실패, 스킵: {e}")
                continue

            for y in years:
                year_path = f"{base}/{y}"
                try:
                    months = list_months(ftp, year_path)
                except Exception as e:
                    print(f"[WARN] {v}/{y} month listing 실패, 스킵: {e}")
                    continue

                for m in months:
                    month_path = f"{year_path}/{m}"
                    try:
                        nc_files = list_nc_files_in_month(ftp, month_path)
                    except Exception as e:
                        print(f"[WARN] {v}/{y}/{m} 파일 listing 실패, 스킵: {e}")
                        continue

                    if not nc_files:
                        # 비어있는 달일 수 있음
                        continue

                    for fname in nc_files:
                        remote_path = f"{month_path}/{fname}"
                        # 로컬은 버전/연/월 구조 그대로 만든다
                        local_path = LOCAL_ROOT / v / y / m / fname
                        remote_local_pairs.append((remote_path, local_path))

    finally:
        ftp.quit()
        print("[INFO] FTP connection closed (listing phase)")

    print(f"[INFO] Total .nc files to download: {len(remote_local_pairs)}")

    if not remote_local_pairs:
        print("[WARN] 다운로드할 파일이 없습니다.")
        return

    # 2) 병렬 다운로드
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DOWNLOAD) as ex:
        futures = [
            ex.submit(download_one, r, lp)
            for (r, lp) in remote_local_pairs
        ]
        for fut in as_completed(futures):
            _ = fut.result()

    print("[DONE] All downloads finished.")


if __name__ == "__main__":
    main()
