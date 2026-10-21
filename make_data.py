import cv2
import os
import csv
import time

# ================== CONFIG ==================
BASE_DIR = "dataset"
CSV_PATH = os.path.join(BASE_DIR, "data.csv")

CAM_ID = 0

# ---- ROI ----
NOZZLE_TIP_X = 300
NOZZLE_TIP_Y = 295
CROP_W = 100
CROP_H = 100

# ---- Thông số in ----
FLOW_RATE = 100
FEED_RATE = 60
Z_OFFSET = 0.2
HOTEND = 200
BED = 60

# ---- LABEL ----
UNIFORMITY = "good"
CONTINUITY = "bad"
ADHESION = "good"

# ---- AUTO CAPTURE ----
CAPTURE_INTERVAL = 1.0  # giây

# ============================================

# ====== TẠO BASE DIR ======
os.makedirs(BASE_DIR, exist_ok=True)

# ====== TẠO FOLDER print-x ======
existing_folders = [
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("print-")
]

if len(existing_folders) > 0:
    ids = [int(d.split("-")[1]) for d in existing_folders]
    folder_id = max(ids) + 1
else:
    folder_id = 0

SAVE_DIR = os.path.join(BASE_DIR, f"print-{folder_id}")
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"[INFO] Saving to folder: {SAVE_DIR}")

# ====== TẠO CSV ======
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "img_path", "timestamp",
            "flow_rate", "feed_rate", "z_offset", "hotend", "bed",
            "nozzle_tip_x", "nozzle_tip_y",
            "uniformity", "continuity", "adhesion",
            "session_id"
        ])

# ====== CAMERA ======
cap = cv2.VideoCapture(CAM_ID)

if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit()

print("[INFO] Camera started")

img_id = 0
last_capture_time = 0

print("W/A/S/D: move ROI | Q: out")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read frame")
        break

    # ===== COPY FRAME để vẽ (QUAN TRỌNG) =====
    display_frame = frame.copy()

    h, w, _ = frame.shape

    # ===== Crop từ FRAME GỐC (clean) =====
    x1 = max(0, NOZZLE_TIP_X)
    y1 = max(0, NOZZLE_TIP_Y)
    x2 = min(w, x1 + CROP_W)
    y2 = min(h, y1 + CROP_H)

    crop = frame[y1:y2, x1:x2]  # <-- dùng frame gốc

    # ===== VẼ CHỈ TRÊN display_frame =====
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cx = x1 + CROP_W // 2
    cy = y1 + CROP_H // 2
    cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)

    # ===== TEXT =====
    cv2.putText(display_frame, f"Top-left: ({x1},{y1})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(display_frame, f"Bottom-right: ({x2},{y2})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(display_frame, f"Center: ({cx},{cy})", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # ===== SHOW =====
    cv2.imshow("Full Frame", display_frame)  # <-- hiển thị bản có vẽ
    cv2.imshow("Nozzle Crop", crop)          # <-- crop sạch

    # ===== AUTO SAVE =====
    current_time = time.time()

    if current_time - last_capture_time >= CAPTURE_INTERVAL:
        timestamp = current_time

        img_name = f"image-{img_id}.jpg"
        img_path = os.path.join(SAVE_DIR, img_name)

        cv2.imwrite(img_path, crop)  # <-- lưu ảnh sạch

        csv_img_path = f"dataset/print-{folder_id}/{img_name}"

        with open(CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                csv_img_path,
                timestamp,
                FLOW_RATE,
                FEED_RATE,
                Z_OFFSET,
                HOTEND,
                BED,
                NOZZLE_TIP_X,
                NOZZLE_TIP_Y,
                UNIFORMITY,
                CONTINUITY,
                ADHESION,
                folder_id
            ])

        print(f"[AUTO SAVED] {csv_img_path}")

        img_id += 1
        last_capture_time = current_time

    # ===== KEY =====
    key = cv2.waitKey(1) & 0xFF

    if key == ord('w'):
        NOZZLE_TIP_Y -= 5
    elif key == ord('s'):
        NOZZLE_TIP_Y += 5
    elif key == ord('a'):
        NOZZLE_TIP_X -= 5
    elif key == ord('d'):
        NOZZLE_TIP_X += 5
    elif key == ord('q'):
        break

# ===== CLEAN =====
cap.release()
cv2.destroyAllWindows()