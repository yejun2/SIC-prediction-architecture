import numpy as np

arr = np.load("/home/yejun/projects/ipiu_2025/cs2smos_v206_sit_npy/raw"
"/20101027.npy")
print(arr)
print("shape :", arr.shape)
print("dtype :", arr.dtype)
print("min :", arr.min(), "max :", arr.max())
