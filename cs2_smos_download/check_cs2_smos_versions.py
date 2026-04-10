import ftplib

FTP_HOST = "ftp.awi.de"
CS2SMOS_ROOT = "/sea_ice/product/cryosat2_smos"

def main():
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login()
    print(f"[INFO] Connected to {FTP_HOST}")

    ftp.cwd(CS2SMOS_ROOT)
    names = ftp.nlst()

    print("\n[INFO] Directories under:", CS2SMOS_ROOT)
    for n in names:
        print("  -", n)

    # 버전 폴더만 필터링 (v + 숫자)
    version_dirs = [n for n in names if n.startswith("v") and n[1:].isdigit()]
    print("\n[INFO] Detected version folders:")
    for v in version_dirs:
        print("  ->", v)

    ftp.quit()

if __name__ == "__main__":
    main()
