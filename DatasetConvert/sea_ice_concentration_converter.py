import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.colors as colors
import matplotlib as mpl

# ───────────────────────────────────────────────
# 🔧 1) 여기만 바꿔주면 됨
nc_file = "/home/yejun/projects/ipiu_2025/nsidc_G02202_downloads/1978/sic_psn25_19781025_n07_v05r00.nc"  # 시각화할 파일 경로
output_png = "/home/yejun/projects/ipiu_2025/nsidc_G02202_downloads/1978/plot_19781025.png"              # 저장될 이미지 경로
# ───────────────────────────────────────────────


def load_sic_variable(ds):
    cand = ["seaice_conc_cdr", "cdr_seaice_conc", "seaice_conc"]
    for name in cand:
        if name in ds.data_vars:
            return ds[name]
    raise ValueError(f"SIC 변수를 찾을 수 없음: {list(ds.data_vars.keys())}")


def main():
    print("[INFO] Loading:", nc_file)
    ds = xr.open_dataset(nc_file)

    sic = load_sic_variable(ds)

    if "time" in sic.dims:
        sic = sic.isel(time=0)

    sic_np = sic.values.astype(float)

    # FillValue 처리
    fill = sic.attrs.get("_FillValue", None)
    if fill is not None:
        sic_np = np.where(sic_np == fill, np.nan, sic_np)

    # 통계 찍어보기 (디버깅용)
    print(
        "[INFO] SIC stats:",
        "min =", np.nanmin(sic_np),
        "max =", np.nanmax(sic_np),
        "mean =", np.nanmean(sic_np),
    )

    # 정상적인 SIC 값은 0~100 (%)
    # Land 는 0% 혹은 NaN, 혹은 음수일 수 있음
    land_mask = np.isnan(sic_np) | (sic_np < 0)

    # 육지 부분은 mask로 가려서 colormap의 "bad" 색(흰색)으로 표시
    sic_ma = np.ma.masked_where(land_mask, sic_np)

    plt.figure(figsize=(7, 8))

    # 🚀 gamma 낮게 → 저농도도 더 밝게 (대비 강화)
    norm = colors.PowerNorm(gamma=0.3, vmin=0, vmax=1)

    # 컬러맵: inferno가 얼음 패턴 보기 좋음
    cmap = mpl.colormaps["inferno"].copy()
    cmap.set_bad(color="white")  # 육지 = 흰색

    im = plt.imshow(sic_ma, origin="lower", cmap=cmap, norm=norm)

    # 15% 해빙선 추가 (겨울/여름 해빙 범위 확인에 유용)
    try:
        plt.contour(
            sic_np,
            levels=[15],
            colors="cyan",
            linewidths=1.0,
            origin="lower",
        )
    except Exception:
        pass

    plt.colorbar(im, label="Sea Ice Concentration (%)")

    plt.title("Sea Ice Concentration", fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=150)
    plt.close()

    print("[INFO] Saved:", output_png)


if __name__ == "__main__":
    main()
