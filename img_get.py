import cv2
import torch
import numpy as np
from PIL import Image


# =============================
# 📷 HÀM CHỤP ẢNH THÔ (không xử lý)
# =============================
def snapshot_and_process_raw(use_gstreamer=False):
    """
    Chụp ảnh từ camera, trả về:
        - ảnh PIL gốc
        - toạ độ (x, y) nozzle tip (tạm thời cố định giữa khung)
    """
    if use_gstreamer:
        # Jetson IMX219
        gst_pipeline = (
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 "
            "! nvvidconv flip-method=0 ! video/x-raw, width=360, height=360, format=BGRx "
            "! videoconvert ! appsink drop=1"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    else:
        # Webcam USB / Laptop
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Không thể mở camera.")
        return None, None, None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("⚠️ Không đọc được khung hình.")
        return None, None, None

    # --- Crop sơ bộ khung chính (giữ vùng in) ---
    # (x, y, w, h) = (140, 0, 360, 360)
    x, y, w, h = 140, 0, 360, 360
    frame_crop = frame[y:y+h, x:x+w]

    # hiển thị preview
    cv2.imshow("Preview - Current Frame", frame_crop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Dừng chương trình theo yêu cầu người dùng.")
        cv2.destroyAllWindows()
        exit(0)

    # Chuyển sang PIL
    img_pil = Image.fromarray(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB))

    # ⚙️ Giả định nozzle_tip nằm ở giữa khung crop
    nozzle_tip_x = frame_crop.shape[1] // 2
    nozzle_tip_y = frame_crop.shape[0] // 2

    return img_pil, nozzle_tip_x, nozzle_tip_y


# =============================
# 📷 HÀM CŨ - GIỮ LẠI CHO CODE KHÁC
# =============================
def snapshot_and_process(model, preprocess_fn, use_gstreamer=False):
    """
    Phiên bản cũ: chụp + xử lý + dự đoán luôn
    (giữ lại cho tương thích code cũ)
    """
    if use_gstreamer:
        gst_pipeline = (
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 "
            "! nvvidconv flip-method=0 ! video/x-raw, width=360, height=360, format=BGRx "
            "! videoconvert ! appsink drop=1"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Không thể mở camera.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("⚠️ Không đọc được khung hình.")
        return None

    x, y, w, h = 140, 0, 360, 360
    frame_crop = frame[y:y+h, x:x+w]

    cv2.imshow("Preview - Current Frame", frame_crop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Dừng chương trình theo yêu cầu người dùng.")
        cv2.destroyAllWindows()
        exit(0)

    img = Image.fromarray(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB))
    x_tensor = preprocess_fn(img).unsqueeze(0).cuda()

    with torch.no_grad():
        preds = model(x_tensor)
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = preds.squeeze().cpu().numpy()

    return preds
