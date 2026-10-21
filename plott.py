import pandas as pd
import itertools

# đọc file
df = pd.read_csv(r"D:\Dung Project\dungvenv\research\caxton_dataset\merged_dataset.csv")

features = ["uniformity", "continuity", "adhesion"]

# -----------------------------
# thống kê tổ hợp có trong dataset
# -----------------------------
combo_counts = (
    df.groupby(features)
      .size()
      .reset_index(name="count")
)

# -----------------------------
# tạo toàn bộ 8 tổ hợp (2^3)
# -----------------------------
all_combos = pd.DataFrame(
    list(itertools.product([0,1], [0,1], [0,1])),
    columns=features
)

# -----------------------------
# merge để hiện cả tổ hợp không tồn tại
# -----------------------------
combo_full = (
    all_combos
    .merge(combo_counts, on=features, how="left")
    .fillna(0)
)

combo_full["count"] = combo_full["count"].astype(int)
combo_full["percent"] = combo_full["count"] / len(df) * 100

print("\nPhân bố đầy đủ 8 tổ hợp:")
print(combo_full.sort_values("count", ascending=False))

# =====================================================
# THỐNG KÊ TỪNG CLASS CỦA MỖI ĐẶC TÍNH
# =====================================================

print("\nThống kê từng đặc tính:")

for f in features:
    
    stat = (
        df[f]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    
    stat.columns = [f, "count"]
    stat["percent"] = stat["count"] / len(df) * 100
    
    print(f"\n--- {f.upper()} ---")
    print(stat)