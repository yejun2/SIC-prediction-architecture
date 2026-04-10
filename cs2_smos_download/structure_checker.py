#!/usr/bin/env python
import ftplib

FTP_HOST = "ftp.awi.de"
CS2SMOS_ROOT = "/sea_ice/product/cryosat2_smos"

TARGET_VERSIONS = ["v204", "v205", "v206"]


def connect_ftp():
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login()
    print(f"[INFO] Connected to {FTP_HOST}")
    return ftp


def list_dir(ftp, path, indent=0):
    """해당 경로의 파일/폴더 목록을 트리 형식으로 출력"""
    prefix = " " * indent
    try:
        ftp.cwd(path)
        items = ftp.nlst()
        print(f"{prefix}[DIR] {path}  ({len(items)} items)")
        for it in items:
            print(f"{prefix}  - {it}")
        print()
    except Exception as e:
        print(f"{prefix}[ERROR] Cannot access {path}: {e}")
        return []


def explore_version(ftp, version):
    print("=" * 80)
    print(f"[INFO] Exploring version: {version}")
    print("=" * 80)

    base = f"{CS2SMOS_ROOT}/{version}"
    list_dir(ftp, base, indent=2)

    # 1) nh 디렉토리 찾기
    try:
        ftp.cwd(base)
        entries = ftp.nlst()
        nh_list = [d for d in entries if d.lower().startswith("nh")]
    except Exception as e:
        print(f"  [ERROR] cannot list NH folder: {e}")
        return

    if not nh_list:
        print("  [WARN] No NH folder found!")
        return

    for nh in nh_list:
        nh_path = f"{base}/{nh}"
        list_dir(ftp, nh_path, indent=4)

        # nh 내부 하위폴더도 탐색 (LATEST or 날짜 기반 폴더 등)
        try:
            ftp.cwd(nh_path)
            inside = ftp.nlst()
        except Exception as e:
            print(f"    [ERROR] cannot list NH subfolders: {e}")
            continue

        for sub in inside:
            sub_path = f"{nh_path}/{sub}"
            # 폴더인지 파일인지 확인 필요
            # FTP는 디렉토리 판별을 지원하지 않으니 try/catch로 판단
            try:
                ftp.cwd(sub_path)  # 디렉토리일 가능성
                list_dir(ftp, sub_path, indent=6)
            except:
                # 파일이면 그냥 표시
                print(f"      [FILE] {sub_path}")


def main():
    ftp = connect_ftp()

    for v in TARGET_VERSIONS:
        explore_version(ftp, v)

    ftp.quit()
    print("\n[DONE] Structure exploration complete.")


if __name__ == "__main__":
    main()
