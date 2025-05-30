from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import threading
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vehicle_model = YOLO('artifacts/yolo11n.pt').to(device)
license_plate_model = YOLO('artifacts/license_plate_detector.pt').to(device)
tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture("samples/bikeincarout.mp4")

lock = threading.Lock()
latest_frame = None
car_count, bike_count = 0, 0
vehicle_directions, track_class_labels, tracked_vehicles = {}, {}, {}

frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
midline = 360

def read_license_plate(frame, bbox):
    """Reads license plate using OCR"""
    x1, y1, x2, y2 = map(int, bbox)
    gray_plate = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray_plate)
    return ' '.join([r[1] for r in results]).strip() if results else None

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
        detections = np.array([[*result.xyxy[0].cpu().numpy(), result.conf[0].cpu().item(), int(result.cls[0].cpu().item())] 
                              for result in vehicle_results[0].boxes])

        if detections.size > 0:
            tracks = tracker.update(detections[:, :5])

            for track in tracks:
                track_id, x1, y1, x2, y2 = int(track[4]), *map(int, track[:4])
                cls = next((det[5] for det in detections if x1 <= det[0] <= x2 and y1 <= det[1] <= y2), None)

                if cls is not None:
                    track_class_labels[track_id] = cls

                center_y = (y1 + y2) // 2
                prev_dir = vehicle_directions.get(track_id)

                if prev_dir is None:
                    vehicle_directions[track_id] = 'up' if center_y < midline else 'down'
                elif prev_dir == 'up' and center_y > midline:
                    if track_class_labels.get(track_id) == 2: car_count += 1
                    elif track_class_labels.get(track_id) == 3: bike_count += 1
                    vehicle_directions[track_id] = None
                elif prev_dir == 'down' and center_y < midline:
                    if track_class_labels.get(track_id) == 2: car_count -= 1
                    elif track_class_labels.get(track_id) == 3: bike_count -= 1
                    vehicle_directions[track_id] = None

                if track_id not in tracked_vehicles:
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size != 0:
                        license_plate_results = license_plate_model(vehicle_crop, conf=0.5, iou=0.4)
                        for lp in license_plate_results[0].boxes:
                            lp_x1, lp_y1, lp_x2, lp_y2 = lp.xyxy[0].cpu().numpy() + [x1, y1, x1, y1]
                            tracked_vehicles[track_id] = read_license_plate(frame, [lp_x1, lp_y1, lp_x2, lp_y2])
                            break

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id} Plate: {tracked_vehicles.get(track_id, 'N/A')}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.line(frame, (0, midline), (frame_width, midline), (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        with lock:
            latest_frame = buffer.tobytes()

threading.Thread(target=process_frames, daemon=True).start()

def generate_frames():
    """Yields video frames for the live stream."""
    while True:
        with lock:
            if latest_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/get_counts")
def get_counts():
    return {"cars": car_count, "bikes": bike_count}
