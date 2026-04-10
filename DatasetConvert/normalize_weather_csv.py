import pandas as pd
import numpy as np

# 입력 CSV
input_csv = "artic_era.csv"
output_csv = "artic_era_normalized.csv"

# weather 컬럼 목록
weather_cols = ["sst_mean", "t2m_mean", "d2m_mean", "tp_mean"]

df = pd.read_csv(input_csv)

# 정규화된 결과 저장할 dictionary
norm_data = {}

for col in weather_cols:
    values = df[col].astype(float)

    # NaN 무시하고 평균/표준편차 계산
    mean = values.mean(skipna=True)
    std = values.std(skipna=True)

    print(f"[INFO] Column {col}: mean={mean:.6f}, std={std:.6f}")

    # NaN은 그대로 두되, 값이 있을 때만 정규화 적용
    norm_col = (values - mean) / std
    norm_col = norm_col.where(~values.isna(), other=np.nan)

    # 새로운 컬럼 추가
    df[col] = norm_col

# 저장
df.to_csv(output_csv, index=False)
print("[INFO] Normalized CSV saved to:", output_csv)
