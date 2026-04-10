import numpy as np

def count_zero_in_mask(mask_path):
    mask = np.load(mask_path)  # (H, W) 또는 (1, H, W)
    
    if mask.ndim == 3:
        mask = mask[0]         # (1, H, W) → (H, W)
    
    # 0 == 바다 (예측 대상)
    zero_count = np.sum(mask == 0)
    one_count  = np.sum(mask == 1)

    print(f"File: {mask_path}")
    print(f"  Zero count (sea)   : {zero_count}")
    print(f"  One count (land)   : {one_count}")
    print(f"  Total pixels       : {mask.size}")
    print(f"  Sea ratio          : {zero_count / mask.size:.4f}")
    print(f"  Land ratio         : {one_count / mask.size:.4f}")

    return zero_count

count_zero_in_mask("/home/yejun/projects/ipiu_2025/edge_maskmap_npy/20210102.npy")
