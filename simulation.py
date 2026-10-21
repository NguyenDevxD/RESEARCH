import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import time
from collections import Counter, defaultdict

from model.network_module import ParametersClassifier
from train_config import DATASET_MEAN, DATASET_STD

# ===== IMPORT FUZZY =====
from fuzzy.controller import fuzzy_controller
from fuzzy.config import clamp
from fuzzy.fuzzy_rules import get_actions

# ======================= CONFIG =======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = r"E:\Research\epoch01-val_acc0.0000.ckpt"

IMAGE_DIM   = 40
NOZZLE_TIP  = (328, 426)
RESIZE_SIZE = 224

PREDICT_INTERVAL = 0.5
WINDOW_SIZE = 10

mean = DATASET_MEAN
std  = DATASET_STD

PARAM_KEYS = ["uniformity", "continuity", "adhesion"]

# ======================= CURRENT PARAM =======================
current_params = {
    "flow_rate": 100,
    "feed_rate": 60,
    "z_offset": 0.2,
    "hotend": 200
}

def to_label(val):
    return "Good" if val == 1 else "Bad"

# ======================= LOAD MODEL =======================
model = ParametersClassifier.load_from_checkpoint(CKPT_PATH)
model.to(DEVICE)
model.eval()

# ======================= DECODE =======================
def decode_output(logits, threshold=0.5):
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    return {
        "uniformity": int(probs[0] > threshold),
        "continuity": int(probs[1] > threshold),
        "adhesion":   int(probs[2] > threshold),
    }

# ======================= LIGHTING =======================
class LightingStabilizer:
    def __init__(self, alpha=0.85, target_v=140):
        self.alpha = alpha
        self.target_v = target_v
        self.prev_frame = None

    def __call__(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        scale = self.target_v / (np.mean(v) + 1e-6)
        scale = np.clip(scale, 0.9, 1.1)
        v = np.clip(v * scale, 0, 255).astype(np.uint8)

        img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

        if self.prev_frame is None:
            self.prev_frame = img.astype(np.float32)
            return img

        smoothed = self.alpha * self.prev_frame + (1 - self.alpha) * img.astype(np.float32)
        self.prev_frame = smoothed
        return smoothed.astype(np.uint8)

lighting_stabilizer = LightingStabilizer()

# ======================= TRANSFORM =======================
transform = transforms.Compose([
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def crop_nozzle(frame):
    cx, cy = NOZZLE_TIP
    d = IMAGE_DIM // 2
    h, w = frame.shape[:2]
    return frame[max(cy-d,0):min(cy+d,h), max(cx-d,0):min(cx+d,w)]

def to_tensor(crop):
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop)
    return transform(img).unsqueeze(0).to(DEVICE)

# ======================= CAMERA =======================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Không mở được camera")

# ======================= BUFFER =======================
buffer = defaultdict(list)
last_predict = 0.0

print("\n▶ Majority voting (10 predictions)")

# ======================= LOOP =======================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    now = time.time()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_stable = lighting_stabilizer(frame.copy())

    # ===== predict =====
    if now - last_predict >= PREDICT_INTERVAL:
        last_predict = now

        crop = crop_nozzle(frame_stable)

        with torch.no_grad():
            logits = model(to_tensor(crop))
            preds = decode_output(logits)

        # append buffer
        for k, v in preds.items():
            buffer[k].append(v)

        # majority voting
        if len(buffer["uniformity"]) >= WINDOW_SIZE:

            def majority(arr):
                return Counter(arr).most_common(1)[0][0]

            result = {k: majority(buffer[k]) for k in PARAM_KEYS}

            print(f"\nMAJORITY → U={result['uniformity']} C={result['continuity']} A={result['adhesion']}")

            # ===== FUZZY CONTROL =====
            u = to_label(result["uniformity"])
            c = to_label(result["continuity"])
            a = to_label(result["adhesion"])

            actions = get_actions(u, c, a)

            new_params = fuzzy_controller(u, c, a, current_params)
            new_params = clamp(new_params)

            # ===== PRINT DEBUG =====
            print(f"Label → U={u}, C={c}, A={a}")
            print(f"Fuzzy Actions → Flow:{actions[0]} | Feed:{actions[1]} | Z:{actions[2]} | Hotend:{actions[3]}")
            print(f"OLD PARAMS → {current_params}")
            print(f"NEW PARAMS → {new_params}")

            # update param (closed-loop)
            current_params = new_params

            # reset buffer
            buffer.clear()

    # ===== display =====
    vis = frame.copy()
    cx, cy = NOZZLE_TIP
    d = IMAGE_DIM // 2
    cv2.rectangle(vis, (cx-d, cy-d), (cx+d, cy+d), (0,255,0), 2)
    cv2.imshow("Majority Voting", vis)

# ======================= CLEAN =======================
cap.release()
cv2.destroyAllWindows()