from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

app = Flask(__name__)

# Enable auto-reloading of templates
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load YOLO models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vehicle_model = YOLO('artifacts/yolo11n.pt').to(device)
license_plate_model = YOLO('artifacts/license_plate_detector.pt').to(device)

# Tracker and OCR
tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)
reader = easyocr.Reader(['en'])

# Video Capture
video_path = "samples/bikeincarout.mp4"
cap = cv2.VideoCapture(video_path)

# Shared Data
latest_frame = None
lock = threading.Lock()
car_count = 0
bike_count = 0
vehicle_directions = {}
track_class_labels = {}
tracked_vehicles = {}
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
midline = 360

def read_license_plate(frame, bbox):
    """Reads license plate using OCR"""
    x1, y1, x2, y2 = map(int, bbox)
    license_plate = frame[y1:y2, x1:x2]
    gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray_plate)
    text = ' '.join([result[1] for result in results])
    return text.strip() if text else None # 'Unreadable'

def process_frames():
    """Processes video frames and tracks vehicles"""
    global latest_frame, car_count, bike_count
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = frame[150:800, :]
        vehicle_results = vehicle_model(frame, conf=0.5, iou=0.4, classes=[2, 3, 5, 7])
        detections = []

        # Extract bounding boxes, confidence, and class info
        for result in vehicle_results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
            conf = result.conf[0].cpu().item()
            cls = int(result.cls[0].cpu().item())
            detections.append([x1, y1, x2, y2, conf, cls])

        if detections:
            dets = np.array(detections)
            tracks = tracker.update(dets[:, :5])

            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                x1, y1, x2, y2 = map(int, bbox)

                cls = None
                for det in detections:
                    det_x1, det_y1, det_x2, det_y2, det_conf, det_cls = det
                    if (x1 <= det_x1 <= x2 and y1 <= det_y1 <= y2) or (x1 <= det_x2 <= x2 and y1 <= det_y2 <= y2):
                        cls = det_cls
                        break

                if cls is not None:
                    track_class_labels[track_id] = cls

                # Detect crossing the midline
                center_y = (y1 + y2) // 2
                if track_id not in vehicle_directions:
                    vehicle_directions[track_id] = None

                if vehicle_directions[track_id] is None:
                    if center_y < midline:
                        vehicle_directions[track_id] = 'up'
                    else:
                        vehicle_directions[track_id] = 'down'
                else:
                    if vehicle_directions[track_id] == 'up' and center_y > midline:
                        if track_class_labels.get(track_id) == 2:
                            car_count += 1
                        elif track_class_labels.get(track_id) == 3:
                            bike_count += 1
                        vehicle_directions[track_id] = None
                    elif vehicle_directions[track_id] == 'down' and center_y < midline:
                        if track_class_labels.get(track_id) == 2:
                            car_count -= 1
                        elif track_class_labels.get(track_id) == 3:
                            bike_count -= 1
                        vehicle_directions[track_id] = None

                # License plate detection
                if track_id not in tracked_vehicles:
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size != 0:
                        license_plate_results = license_plate_model(vehicle_crop, conf=0.5, iou=0.4)
                        for lp_result in license_plate_results[0].boxes:
                            lp_x1, lp_y1, lp_x2, lp_y2 = lp_result.xyxy[0].cpu().numpy()
                            lp_x1 += x1
                            lp_y1 += y1
                            lp_x2 += x1
                            lp_y2 += y1

                            license_text = read_license_plate(frame, [lp_x1, lp_y1, lp_x2, lp_y2])
                            tracked_vehicles[track_id] = license_text
                            break

                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id} Plate: {tracked_vehicles.get(track_id, 'N/A')}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.line(frame, (0, midline), (frame_width, midline), (255, 0, 0), 2)
        # cv2.putText(frame, f"Cars: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, f"Bikes: {bike_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        with lock:
            latest_frame = buffer.tobytes()

threading.Thread(target=process_frames, daemon=True).start()

def generate_frames():
    """Yields video frames"""
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def home():
    """Serves the dashboard template"""
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """Serves the live video stream"""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_counts")
def get_counts():
    """Returns live vehicle counts"""
    return jsonify({"cars": car_count, "bikes": bike_count})

if __name__ == "__main__":
    app.run(debug=True)
