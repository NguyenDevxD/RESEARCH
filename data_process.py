'''
import pandas as pd
import itertools

dataset = pd.read_csv(r"D:\Dung Project\dungvenv\research\filtered.csv")
mapping = pd.read_csv("features_mapping.csv")

df = dataset.merge(
    mapping,
    on=[
        "flow_rate_class",
        "feed_rate_class",
        "z_offset_class",
        "hotend_class"
    ],
    how="inner"
)

df.to_csv("suplement_data.csv",index=False)

import pandas as pd

df = pd.read_csv(r"D:\Dung Project\dungvenv\research\full_data.csv")

features = ["filament_width", "filament_continuity", "filament_adhesion"]

TARGET = 110000

dfs = []

for combo, group in df.groupby(features):

    if len(group) > TARGET:
        group = group.sample(n=TARGET, random_state=42)

    dfs.append(group)

df_balanced = pd.concat(dfs).reset_index(drop=True)

print("Dataset mới:", len(df_balanced))

# kiểm tra lại phân bố
print("\nPhân bố tổ hợp mới:")
print(df_balanced.groupby(features).size())

df_balanced.to_csv("balanced_dataset.csv", index=False)
'''
import pandas as pd

df = pd.read_csv(r"D:\Dung Project\dungvenv\research\data.csv")

# các state cần giảm
states_to_reduce = [
    (1,1,1),
]

# tỉ lệ giữ lại (ví dụ 50%)
keep_ratio = 0.6

kept_parts = []

for u,c,a in states_to_reduce:

    mask = (
        (df["uniformity"] == u) &
        (df["continuity"] == c) &
        (df["adhesion"] == a)
    )

    subset = df[mask]

    print(f"Original ({u},{c},{a}):", len(subset))

    sampled = subset.sample(frac=keep_ratio, random_state=42)

    print(f"Remaining ({u},{c},{a}):", len(sampled))

    kept_parts.append(sampled)

# giữ nguyên các state khác
mask_reduce = df.apply(
    lambda r: (r["uniformity"],r["continuity"],r["adhesion"]) in states_to_reduce,
    axis=1
)

others = df[~mask_reduce]

# ghép lại
df_new = pd.concat([others] + kept_parts, ignore_index=True)

# shuffle dataset
df_new = df_new.sample(frac=1, random_state=42).reset_index(drop=True)

df_new.to_csv("balanced_data.csv", index=False)

print("Final dataset size:", len(df_new))