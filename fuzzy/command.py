import cv2
import torch
import time
from model.network_module import ParametersRegressor
from train_config import preprocess
from fuzzy_system import FuzzySystem

# =============================
# 🔧 CONFIG
# =============================
checkpoint = r"D:\computer vision\opencv\myvenv\DATN 2025.1\parameters_analysis_model\checkpoints\MHResAttNet-dataset_full-07102025-best-epoch=09-val_loss=0.0075.ckpt"
checkpoint = checkpoint.replace("\\", "/")

# True nếu dùng camera IMX219 trên Jetson
USE_GSTREAMER = False

# =============================
# 🧠 LOAD MODEL
# =============================
print("🔄 Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ParametersRegressor.load_from_checkpoint(
    checkpoint_path=checkpoint,
    map_location=device
)
model.eval().to(device)
print("✅ Model loaded successfully from", checkpoint)

# =============================
# 🤖 KHỞI TẠO FUZZY SYSTEM
# =============================
fs_flow = FuzzySystem("flow_rate")
fs_feed = FuzzySystem("feed_rate")
fs_z    = FuzzySystem("z_offset")
fs_temp = FuzzySystem("hotend")

# =============================
# ⚙️ THÔNG SỐ BAN ĐẦU
# =============================
flow_last = 157
feed_last = 140
z_last    = -0.05
temp_last = 220.98

# =============================
# 📷 KHỞI TẠO CAMERA
# =============================
print("📷 Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Không thể mở webcam!")

# Thiết lập độ phân giải
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =============================
# 🚀 MAIN LOOP
# =============================
print("Bắt đầu chạy vòng lặp, nhấn 'q' để thoát.\n")
print("step\tdev_flow\tadj_flow\tflow\tdev_feed\tadj_feed\tfeed\tdev_z\tadj_z\tz\tdev_temp\tadj_temp\ttemp")

i = 1
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Không nhận được khung hình từ webcam.")
        continue

    # ====== HIỂN THỊ ẢNH ======
    cv2.imshow("Webcam (press q to quit)", frame)

    # ====== XỬ LÝ ẢNH ======
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = preds.squeeze().detach().cpu().numpy()

    # unpack 4 deviation
    dev_flow, dev_feed, dev_z, dev_temp = preds

    # ====== FUZZY INFERENCE ======
    adj_flow = fs_flow.infer(dev_flow)
    adj_feed = fs_feed.infer(dev_feed)
    adj_z    = fs_z.infer(dev_z)
    adj_temp = fs_temp.infer(dev_temp)

    # ====== CẬP NHẬT GIÁ TRỊ MỚI ======
    flow_new = flow_last + adj_flow
    feed_new = feed_last + adj_feed
    z_new    = z_last + adj_z
    temp_new = temp_last + adj_temp

    # ====== IN KẾT QUẢ RA MÀN HÌNH ======
    print(f"{i}\t{dev_flow:.3f}\t{adj_flow:+.3f}\t{flow_new:.3f}\t"
          f"{dev_feed:.3f}\t{adj_feed:+.3f}\t{feed_new:.3f}\t"
          f"{dev_z:.3f}\t{adj_z:+.3f}\t{z_new:.3f}\t"
          f"{dev_temp:.3f}\t{adj_temp:+.3f}\t{temp_new:.2f}")

    # ====== CẬP NHẬT TRẠNG THÁI ======
    flow_last, feed_last, z_last, temp_last = flow_new, feed_new, z_new, temp_new

    # ====== THOÁT ======
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =============================
# 🧹 CLEANUP
# =============================
cap.release()
cv2.destroyAllWindows()
print("🛑 Dừng chương trình.")
