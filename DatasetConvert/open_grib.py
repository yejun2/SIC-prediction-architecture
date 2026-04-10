# /home/yejun/projects/ipiu_2025/open_grib.py

import cfgrib
import pandas as pd

grib_path = "/home/yejun/projects/ipiu_2025/7ffa054aa6adc575e2eda6bc5fad7b0.grib"
output_csv = "./artic_era.csv"

# 북극 영역 (60N~90N)
LAT_MAX = 90.0
LAT_MIN = 60.0

# CSV에 넣고 싶은 변수 목록
# key: CSV에 쓸 변수 이름 prefix, val: GRIB 안의 실제 변수 이름
VAR_LIST = {
    "sst": "sst",   # sea surface temperature
    "t2m": "t2m",   # 2m temperature
    "d2m": "d2m",   # 2m dewpoint
    "tp": "tp",     # total precipitation
    # "mwd": "mwd", # 필요하면 주석 해제해서 mean wave direction도 포함
}


def find_var(ds_list, var_name):
    """
    open_datasets로 연 ds_list에서 주어진 변수(var_name)를 찾아서
    (dataset, DataArray)를 반환. 못 찾으면 (None, None)
    """
    for ds in ds_list:
        if var_name in ds.data_vars:
            return ds, ds[var_name]
    return None, None


def main():
    # 1) GRIB 파일을 여러 Dataset으로 모두 연다
    ds_list = cfgrib.open_datasets(grib_path)

    frames = []

    for csv_key, grib_var in VAR_LIST.items():
        print(f"\n--- Processing {csv_key} (GRIB var='{grib_var}') ---")

        ds, da = find_var(ds_list, grib_var)
        if da is None:
            print(f"[WARN] Variable '{grib_var}' not found in any dataset. Skip.")
            continue

        # 2) 위도 범위 슬라이싱 (latitude가 있을 때만)
        if "latitude" in da.coords:
            da_sel = da.sel(latitude=slice(LAT_MAX, LAT_MIN))
            print(
                f"{csv_key}: lat range "
                f"{float(da_sel.latitude.max())} → {float(da_sel.latitude.min())}"
            )
        else:
            da_sel = da
            print(f"{csv_key}: no explicit latitude coord, using whole field as-is.")

        # 3) 시간 좌표 선택: time 우선, 없으면 valid_time 사용
        time_coord = None
        for cand in ["time", "valid_time"]:
            if (cand in da_sel.coords) or (cand in da_sel.dims):
                time_coord = cand
                break

        if time_coord is None:
            print(f"[WARN] No usable time coordinate for {csv_key}. Skip.")
            continue

        # 4) 공간 평균: latitude/longitude만 평균 대상으로 사용
        spatial_dims = [d for d in da_sel.dims if d in ("latitude", "longitude")]
        if spatial_dims:
            da_mean = da_sel.mean(dim=tuple(spatial_dims))
            print(f"{csv_key}: mean over spatial dims {spatial_dims} → dims={da_mean.dims}")
        else:
            da_mean = da_sel
            print(f"{csv_key}: no spatial dims to average, using as-is with dims={da_mean.dims}")

        # 5) DataFrame으로 변환 (value 컬럼 이름을 명시적으로 지정)
        value_col = f"{csv_key}_mean"
        df = da_mean.to_dataframe(name=value_col).reset_index()

        if time_coord not in df.columns:
            print(
                f"[WARN] time coordinate '{time_coord}' not in DataFrame columns for {csv_key}. "
                f"Got columns: {df.columns}"
            )
            continue

        # time/valid_time 중 무엇을 썼든 최종 CSV에선 'time'으로 통일
        df = df[[time_coord, value_col]].rename(columns={time_coord: "time"})

        frames.append(df)

    # 6) 모든 변수 time 기준으로 outer merge
    if not frames:
        raise RuntimeError("No variables extracted successfully.")

    df_final = frames[0]
    for df in frames[1:]:
        df_final = df_final.merge(df, on="time", how="outer")

    df_final = df_final.sort_values(by="time")
    df_final.to_csv(output_csv, index=False)

    print("\nSaved:", output_csv)
    print(df_final.head())


if __name__ == "__main__":
    main()
