import numpy as np
from pathlib import Path
from skimage.transform import resize  # pip install scikit-image

# 786x786 마스크 npy 경로
MASK_786_PATH = Path("/home/yejun/projects/ipiu_2025/tmp_data/edge_maskmap_npy/20210102.npy")

# 432x432로 줄인 마스크 저장 경로
OUT_MASK_432_PATH = Path("/home/yejun/projects/ipiu_2025/mask_432x432.npy")


def main():
    mask_786 = np.load(MASK_786_PATH)  # (786, 786)

    print("[INFO] original mask shape:", mask_786.shape)

    # 마스크값이 0/1 또는 0/255 이런 정수일 거라 가정
    # nearest neighbor resize (order=0, anti_aliasing=False)
    mask_432 = resize(
        mask_786,
        (432, 432),
        order=0,              # nearest
        preserve_range=True,  # 값 범위 그대로 유지
        anti_aliasing=False,
    ).astype(mask_786.dtype)

    print("[INFO] resized mask shape:", mask_432.shape)

    np.save(OUT_MASK_432_PATH, mask_432)
    print("[INFO] saved:", OUT_MASK_432_PATH)


if __name__ == "__main__":
    main()
