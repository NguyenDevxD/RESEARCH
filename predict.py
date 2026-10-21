import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model.network_module import ParametersClassifier
from train_config import *
import matplotlib.pyplot as plt

# ======================= CONFIG =======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = r"D:\Dung Project\dungvenv\research\epoch01-val_acc0.0000.ckpt"
IMG_PATH  = r"E:\DATN\caxton_dataset\z7633347786512_4afdf73daaa6ddba4e227b33ce965afd.jpg"

IMAGE_DIM   = 140
NOZZLE_TIP  = (691, 854)
RESIZE_SIZE = 224

mean = DATASET_MEAN
std  = DATASET_STD

# ======================= LOAD MODEL =======================
model = ParametersClassifier.load_from_checkpoint(CKPT_PATH)
model.to(DEVICE)
model.eval()

# ======================= HELPER =======================
def decode_output(logits):
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    prob_u, prob_c, prob_a = probs

    return {
        "uniformity": (int(prob_u > 0.5), prob_u),
        "continuity": (int(prob_c > 0.5), prob_c),
        "adhesion":   (int(prob_a > 0.5), prob_a),
    }

def print_result(result):
    for k, (pred, prob) in result.items():
        label = "GOOD" if pred == 1 else "BAD"
        print(f"{k:12s}: {label} ({prob:.3f})")

# ======================================================
# 🔥 MATPLOTLIB VISUAL DEBUG
# ======================================================
def show_with_crop_matplotlib(frame):

    cx, cy = NOZZLE_TIP
    d = IMAGE_DIM // 2

    x1, y1 = cx - d, cy - d
    x2, y2 = cx + d, cy + d

    crop = frame[y1:y2, x1:x2]

    # convert BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crop_rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # ===== FULL IMAGE =====
    axs[0].imshow(frame_rgb)
    axs[0].set_title("FULL IMAGE")

    # vẽ bounding box
    rect = plt.Rectangle(
        (x1, y1),
        IMAGE_DIM,
        IMAGE_DIM,
        fill=False,
        linewidth=2
    )
    axs[0].add_patch(rect)

    # vẽ tâm
    axs[0].scatter(cx, cy)

    # hiển thị tọa độ
    axs[0].text(cx, cy, f"({cx},{cy})")

    # ===== CROP =====
    axs[1].imshow(crop_rgb)
    axs[1].set_title(f"CROP\n({x1},{y1}) → ({x2},{y2})")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# ======================= PREPROCESS =======================
transform = transforms.Compose([
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def to_tensor(frame):
    cx, cy = NOZZLE_TIP
    d = IMAGE_DIM // 2

    crop = frame[cy-d:cy+d, cx-d:cx+d]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(crop)
    return transform(img).unsqueeze(0).to(DEVICE)

# ======================= RUN =======================
frame = cv2.imread(IMG_PATH)

print("\n========= TEST =========")

# 🔥 show debug trước khi infer
show_with_crop_matplotlib(frame)

# ===== inference =====
x = to_tensor(frame)

with torch.no_grad():
    logits = model(x)

result = decode_output(logits)

print_result(result)