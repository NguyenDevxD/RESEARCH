import torch
import time
import cv2
import os
from PIL import Image
from torchvision import transforms

from finetune_regressor import ParametersRegressor
from train_config import preprocess

# Nếu bạn có fuzzy system thật:
try:
    from Code.fuzzy_system import FuzzySystem
except ImportError:
    print("⚠️ Không tìm thấy fuzzy_system, dùng bản giả lập tạm thời.")
    class FuzzySystem:
        def __init__(self, name):
            self.name = name
        def infer(self, x):
            return -0.1 * x

# =============================
# CONFIG
# =============================
checkpoint = r"D:/computer vision/opencv/myvenv/DATN 2025.1/parameters_analysis_model/checkpoints/MHResAttNet-dataset_full-07102025-best-epoch=09-val_loss=0.0075.ckpt"
checkpoint = checkpoint.replace("\\", "/")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

print("⏳ Loading model...")
model = ParametersRegressor.load_from_checkpoint(
    checkpoint_path=checkpoint,
    map_location=device
)
model.eval().to(device)
print("✅ Model loaded successfully from", checkpoint)

# Fuzzy systems
fs_flow = FuzzySystem("flow_rate")
fs_feed = FuzzySystem("feed_rate")
fs_z    = FuzzySystem("z_offset")
fs_temp = FuzzySystem("hotend")

# =============================
# 📸 USB CAMERA SETUP
# =============================

use_gstreamer = False  # ✅ BẮT BUỘC ĐỂ FALSE ĐỂ DÙNG WEBCAM USB

if use_gstreamer:
    print("❌ GStreamer đang bật — chuyển sang use_gstreamer = False để dùng USB camera.")
    exit()
else:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # ✅ USB cam

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Không thể mở USB camera.")
    exit()

print("✅ USB camera opened")


# =============================
# Crop như dataset
# =============================
def crop_like_dataset(img_pil, nozzle_x, nozzle_y, image_dim=(320, 320)):
    dim = image_dim[0] / 2
    left = int(nozzle_x - dim)
    top = int(nozzle_y - dim)
    right = int(nozzle_x + dim)
    bottom = int(nozzle_y + dim)

    w, h = img_pil.size
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)

    return img_pil.crop((left, top, right, bottom))


# =============================
# MAIN LOOP
# =============================
def main():
    print("📡 Bắt đầu... Nhấn 'q' để thoát")
    i = 1
    interval = 5.0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Không đọc được frame camera")
            continue

        display_frame = frame.copy()

        if time.time() - start >= interval:
            start = time.time()
            print(f"\n🌀 Loop {i}")

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            w, h = img_pil.size
            nozzle_x, nozzle_y = w // 2, h // 2

            img_crop = crop_like_dataset(img_pil, nozzle_x, nozzle_y)
            img_tensor = preprocess(img_crop).unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(img_tensor).squeeze().tolist()

            F_R_pred, L_S_pred, Z_off_pred, H_T_pred = preds

            print(f"🔍 Pred {i}: Flow={F_R_pred:.3f}, Feed={L_S_pred:.3f}, Z={Z_off_pred:.3f}, Temp={H_T_pred:.3f}")

            adj_flow = fs_flow.infer(F_R_pred)
            adj_feed = fs_feed.infer(L_S_pred)
            adj_z    = fs_z.infer(Z_off_pred)
            adj_temp = fs_temp.infer(H_T_pred)

            print(f"🧠 Adjust: ΔFlow={adj_flow:.3f}, ΔFeed={adj_feed:.3f}, ΔZ={adj_z:.3f}, ΔTemp={adj_temp:.3f}")

            cv2.putText(display_frame, f"Flow: {F_R_pred:.3f} Δ{adj_flow:.3f}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
            cv2.putText(display_frame, f"Feed: {L_S_pred:.3f} Δ{adj_feed:.3f}", (20,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
            cv2.putText(display_frame, f"Z: {Z_off_pred:.3f} Δ{adj_z:.3f}", (20,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255),2)
            cv2.putText(display_frame, f"Temp: {H_T_pred:.3f} Δ{adj_temp:.3f}", (20,130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

            cv2.rectangle(display_frame,
                          (nozzle_x-160, nozzle_y-160),
                          (nozzle_x+160, nozzle_y+160),
                          (0,255,0),2)

            i += 1

        cv2.imshow("3D Print Monitor (USB)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Thoát")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
