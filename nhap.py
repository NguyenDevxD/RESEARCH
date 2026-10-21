import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

csv_file = r"D:\Dung Project\dungvenv\research\caxton_dataset\full_data.csv"
root_dir = r"D:\Dung Project\dungvenv\research"

# đọc csv
df = pd.read_csv(csv_file)

# sample để chạy nhanh
df = df.sample(70000, random_state=42).reset_index(drop=True)

resize = transforms.Resize((224, 224))

mean = np.zeros(3)
std = np.zeros(3)
n_pixels = 0

for _, row in tqdm(df.iterrows(), total=len(df)):

    img_path = os.path.join(root_dir, row["img_path"])

    try:
        img = Image.open(img_path).convert("RGB")

        # ===== crop nozzle giống dataset.py =====
        dim = 320 / 2
        left = row["nozzle_tip_x"] - dim
        top = row["nozzle_tip_y"] - dim
        right = row["nozzle_tip_x"] + dim
        bottom = row["nozzle_tip_y"] + dim

        img = img.crop((left, top, right, bottom))

        # resize giống training
        img = resize(img)

        img = np.array(img) / 255.0

        h, w, c = img.shape
        n_pixels += h * w

        mean += img.sum(axis=(0,1))
        std += (img**2).sum(axis=(0,1))

    except:
        continue

mean /= n_pixels
std = np.sqrt(std / n_pixels - mean**2)

print("\nDATASET_MEAN =", mean.tolist())
print("DATASET_STD  =", std.tolist())