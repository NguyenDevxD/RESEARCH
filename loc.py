import pandas as pd

# ================= Đọc dữ liệu =================
full_df = pd.read_csv(r"D:\Dung Project\dungvenv\research\caxton_dataset\full_data.csv")
filtered_df = pd.read_csv(r"D:\Dung Project\dungvenv\research\caxton_dataset\supplement_data.csv")

# ================= Clean =================
for df in [full_df, filtered_df]:
    df["img_path"] = df["img_path"].str.strip()

# ================= Remove trùng =================
# chỉ giữ sample trong filtered mà không có trong full
full_paths = set(full_df["img_path"])
filtered_unique = filtered_df[~filtered_df["img_path"].isin(full_paths)]

print(f"Full data: {len(full_df)}")
print(f"Filtered unique: {len(filtered_unique)}")

# ================= Merge =================
merged_df = pd.concat([full_df, filtered_unique], ignore_index=True)

# ================= Shuffle =================
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ================= Save =================
merged_df.to_csv("merged_dataset.csv", index=False)

print(f"Final dataset: {len(merged_df)}")