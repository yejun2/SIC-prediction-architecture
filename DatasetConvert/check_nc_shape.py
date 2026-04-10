import xarray as xr

path = "/home/yejun/projects/ipiu_2025/cs2smos_by_ym/v206/2010/10/W_XX-ESA,SMOS_CS2,NH_25KM_EASE2_20101021_20101027_r_v206_01_l4sit.nc"
ds = xr.open_dataset(path)
print(ds)

