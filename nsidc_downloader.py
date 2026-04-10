import os
import re
import argparse
from urllib.parse import urljoin

import requests


def list_files_in_directory(url, exts=None):
    """
    주어진 NSIDC 디렉토리 URL에서 파일 리스트를 가져온다.
    exts: ['.nc', '.nc.gz'] 같이 확장자 필터. None이면 모두.
    """
    print(f"[INFO] Listing files from: {url}")
    resp = requests.get(url)
    resp.raise_for_status()

    # href="..." 패턴만 대충 긁어오기 (디렉토리 인덱스 HTML이라 이걸로 충분함)
    hrefs = re.findall(r'href="([^"]+)"', resp.text)

    files = []
    for href in hrefs:
        # 상위 디렉토리 링크 등 스킵
        if href in ("../", "/"):
            continue
        # 서브디렉토리는 여기서는 스킵 (필요하면 재귀적으로 처리 가능)
        if href.endswith("/"):
            continue

        # 확장자 필터
        if exts is not None:
            if not any(href.endswith(ext) for ext in exts):
                continue

        full_url = urljoin(url, href)
        files.append((href, full_url))

    print(f"[INFO] Found {len(files)} files.")
    return files


def download_file(url, out_path, chunk_size=1024 * 1024):
    """
    url에서 out_path로 파일 하나 다운로드
    """
    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} (already exists)")
        return

    print(f"[DOWN] {url} -> {out_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # keep-alive chunk 제외
                    f.write(chunk)
    print(f"[DONE] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="다운받고 싶은 연도 (예: 1978)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="nsidc_G02202_downloads",
        help="파일을 저장할 로컬 디렉토리",
    )
    args = parser.parse_args()

    base_url_template = "https://noaadata.apps.nsidc.org/NOAA/G02202_V5/north/daily/{year}/"
    year_url = base_url_template.format(year=args.year)

    # 원하는 확장자만 다운로드 (필요하면 수정 가능)
    exts = [".nc", ".nc.gz"]

    files = list_files_in_directory(year_url, exts=exts)

    if not files:
        print("[WARN] 다운로드할 파일이 없습니다. URL/연도를 다시 확인하세요.")
        return

    for fname, url in files:
        out_path = os.path.join(args.outdir, str(args.year), fname)
        try:
            download_file(url, out_path)
        except Exception as e:
            print(f"[ERROR] {fname} 다운로드 실패: {e}")


if __name__ == "__main__":
    main()
