import torch
import cv2
import numpy as np
import pyttsx3
import easyocr
import socket
import threading
import queue
import json
import asyncio
import websockets
from torchvision.transforms import functional as F
import csv
from datetime import datetime
import time

# ======= Global Variables =======
video_processing_enabled = False
video_processing_lock = threading.Lock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

UDP_IP = "192.168.7.80"
SEND_PORT = 5005
UDP_PORT = 5006
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

selected_mode = "Assistance"
detected_speed = "N/A"
current_motor_speed = "N/A"

engine = pyttsx3.init()
engine.setProperty('rate', 250)
engine.setProperty('volume', 6.0)

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

model_path = "C:/Users/ASUS/Downloads/road_best_model.pt"
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

CLASS_LABELS = {0: "Traffic Light", 1: "Speed Limit", 2: "Crosswalk", 3: "Stop"}
FOCUSED_CLASSES = {"Speed Limit", "Stop"}
VALID_SPEED_VALUES = {str(i) for i in range(10, 121, 5)}

#cap = cv2.VideoCapture(0)
# Open the video source
cap = cv2.VideoCapture("http://192.168.7.80:8080/?action=stream")
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
output_path = "output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_queue = queue.Queue(maxsize=1)
frame_lock = threading.Lock()
latest_frame = None

LOG_FILE = "traffic_logs.csv"
stop_cooldown = False
last_stop_time = 0

try:
    with open(LOG_FILE, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Detected Sign", "Detected Speed", "Motor Speed", "Mode", "Violation"])
except FileExistsError:
    pass

# ======= Violation Log File =======
try:
    with open("violations.csv", 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Detected Sign", "Detected Speed", "Motor Speed", "Mode", "Violation"])
except FileExistsError:
    pass

# ======= WebSocket Server =======
# ... [ALL YOUR EXISTING IMPORTS AND SETUP CODE REMAINS UNCHANGED] ...

# ======= WebSocket Server =======
async def handle_websocket(websocket, path):
    global selected_mode, detected_speed, current_motor_speed, video_processing_enabled
    print("🔗 Client connected!")

    try:
        async for message in websocket:
            if message == "RequestStatus":
                violation_status = "No"
                try:
                    with open(LOG_FILE, 'r') as f:
                        rows = list(csv.reader(f))
                        if len(rows) > 1:
                            last_row = rows[-1]
                            violation_status = last_row[-1]
                except Exception as e:
                    print("⚠️ Error reading log file for violation status:", e)

                await websocket.send(json.dumps({
                    "mode": selected_mode,
                    "detected_speed": detected_speed,
                    "current_speed": current_motor_speed,
                    "video_status": "Running" if video_processing_enabled else "Stopped",
                    "violation": violation_status
                }))
                continue

            print(f"📩 Received from client: {message}")

            if message in ["Autonomous", "Assistance"]:
                selected_mode = message
                sock.sendto(f"Mode:{selected_mode}".encode(), (UDP_IP, SEND_PORT))
                await websocket.send(f"Mode updated: {selected_mode}")

            elif message == "StartVideo":
                with video_processing_lock:
                    video_processing_enabled = True
                await websocket.send("Video processing started.")

            elif message == "StopVideo":
                with video_processing_lock:
                    video_processing_enabled = False
                await websocket.send("Video processing stopped.")

            elif message == "EngineStart":
                sock.sendto("EngineStart".encode(), (UDP_IP, SEND_PORT))
                print("➡ Sending 'EngineStart' to Raspberry Pi")
                await websocket.send("Engine started.")

            elif message == "EngineStop":
                sock.sendto("EngineStop".encode(), (UDP_IP, SEND_PORT))
                print("➡ Sending 'EngineStop' to Raspberry Pi")
                await websocket.send("Engine stopped.")

            else:
                await websocket.send("❌ Invalid command")
    except websockets.exceptions.ConnectionClosed:
        print("🔴 Client disconnected")

# ... [EVERYTHING ELSE IN YOUR CODE REMAINS UNCHANGED] ...


def run_websocket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(handle_websocket, "0.0.0.0", 8765)
    loop.run_until_complete(start_server)
    loop.run_forever()

# ======= Frame Capture Thread =======
def capture_frames():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            latest_frame = frame
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

# ======= TTS =======
def voice_alert(message):
    engine.say(message)
    engine.runAndWait()

# ======= OCR Speed Extraction =======
def extract_speed_limit(frame, box):
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    for res in results:
        numbers = ''.join(filter(str.isdigit, res[1]))
        if numbers in VALID_SPEED_VALUES:
            return numbers
    return None

# ======= Motor Control =======
def control_motor_speed(sign_label, speed_value=None):
    global selected_mode, detected_speed, current_motor_speed
    message = None
    violation_occurred = False
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        motor_speed_kmh = int(current_motor_speed.replace(" km/h", ""))
    except:
        motor_speed_kmh = 0  # Fallback/default

    if sign_label == "Speed Limit" and speed_value:
        detected_speed = f"{speed_value} km/h"
        print(f"⚠ Speed Limit {speed_value} detected.")
        voice_alert(f"Speed limit {speed_value} detected.")
        if selected_mode == "Autonomous":
            message = f"SpeedLimit:{speed_value}"
        if selected_mode == "Assistance" and motor_speed_kmh > int(speed_value):
            violation_occurred = True
            print("🚨 Speed Violation!")

    elif sign_label == "Stop":
        detected_speed = "Stop sign"
        print("🛑 Stop sign detected.")
        voice_alert("Stop.")
        if selected_mode == "Autonomous":
            message = "Stop"
        # Check violation: motor should be at 0 km/h for Stop sign
        if selected_mode == "Assistance" and motor_speed_kmh > 0:
            violation_occurred = True
            print("🚨 Stop Sign Violation!")

    # ===== Log to both files =====
    log_row = [
        timestamp, sign_label, detected_speed, f"{motor_speed_kmh} km/h",
        selected_mode, "Yes" if violation_occurred else "No"
    ]

    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow(log_row)

    if violation_occurred:
        with open("violations.csv", 'a', newline='') as vf:
            csv.writer(vf).writerow(log_row)

    # ===== Send command if needed =====
    if message:
        sock.sendto(message.encode(), (UDP_IP, SEND_PORT))


# ======= Frame Processor =======
def process_frames():
    global latest_frame, stop_cooldown, last_stop_time
    while True:
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        with video_processing_lock:
            if not video_processing_enabled:
                continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(image_tensor)

        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        detected_signs = set()
        for box, score, label in zip(boxes, scores, labels):
            if score < 0.5:
                continue
            class_name = CLASS_LABELS.get(label, "Unknown")
            if class_name not in FOCUSED_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if class_name == "Speed Limit":
                speed_value = extract_speed_limit(frame, box)
                if speed_value:
                    control_motor_speed(class_name, speed_value)

            elif class_name == "Stop" and class_name not in detected_signs:
                if selected_mode == "Autonomous" and not stop_cooldown:
                    control_motor_speed(class_name)
                    stop_cooldown = True
                    last_stop_time = time.time()
                elif selected_mode == "Assistance":
                    control_motor_speed(class_name)
                detected_signs.add(class_name)

        if stop_cooldown and time.time() - last_stop_time > 15:
            stop_cooldown = False

        with frame_lock:
            latest_frame = frame

# ======= Motor Speed Receiver (Separate Thread) =======
def receive_motor_speed():
    global current_motor_speed
    udp_receive_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_receive_sock.bind(("0.0.0.0", UDP_PORT))
    udp_receive_sock.settimeout(1)
    while True:
        try:
            data, _ = udp_receive_sock.recvfrom(1024)
            message = data.decode()
            if message.startswith("CurrentSpeed:"):
                speed = message.split(":")[1]
                current_motor_speed = speed + " km/h"
        except socket.timeout:
            continue

# ======= Start Threads =======
threading.Thread(target=capture_frames, daemon=True).start()
threading.Thread(target=process_frames, daemon=True).start()
threading.Thread(target=receive_motor_speed, daemon=True).start()
threading.Thread(target=run_websocket_server, daemon=True).start()

# ======= Display Loop =======
while True:
    with frame_lock:
        with video_processing_lock:
            if latest_frame is not None and video_processing_enabled:
                cv2.imshow("Traffic Sign Detection", latest_frame)
                cv2.setWindowProperty("Traffic Sign Detection", cv2.WND_PROP_TOPMOST, 1)
                out.write(latest_frame)
            elif not video_processing_enabled:
                try:
                    if cv2.getWindowProperty("Traffic Sign Detection", cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.destroyWindow("Traffic Sign Detection")
                except cv2.error:
                    pass
                cv2.destroyAllWindows()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
sock.close()
