from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)

camera_index = 1  # Thay 0,1,2 tùy camera USB của bạn
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Dùng DSHOW trên Windows

# Thiết lập độ phân giải
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Kiểm tra camera
if not cap.isOpened():
    print(f"❌ Không thể mở camera {camera_index}")
    exit()

print(f"✅ Camera {camera_index} mở thành công")

# Biến lưu frame mới nhất
frame = None

# Thread liên tục đọc frame
def update_frame():
    global frame
    while True:
        ret, f = cap.read()
        if ret:
            frame = f

threading.Thread(target=update_frame, daemon=True).start()

# Hàm generate frame MJPEG
def generate_frames():
    global frame
    while True:
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route stream video
@app.route('/video.mjpg')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route snapshot
@app.route('/snapshot.jpg')
def snapshot():
    global frame
    if frame is None:
        return "Camera error", 500
    ret, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    print("📷 Camera server running at http://0.0.0.0:8081/video.mjpg")
    app.run(host='0.0.0.0', port=8081, threaded=True)
