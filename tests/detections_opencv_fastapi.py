import cv2
import torch
import numpy as np
import threading
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from sort.sort import Sort
import easyocr

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")

vehicle_model = YOLO('artifacts/yolo11n.pt').to(device)
license_plate_model = YOLO('artifacts/license_plate_detector.pt').to(device)
tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)

reader = easyocr.Reader(['en'])

car_count = 0
bike_count = 0
vehicle_directions = {}
track_class_labels = {}
tracked_vehicles = {}

midline = 360
latest_frame = None  # Stores the latest processed frame

def read_license_plate(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    license_plate = frame[y1:y2, x1:x2]
    gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray_plate)
    text = ' '.join([result[1] for result in results])
    return text.strip() if text else None

def process_video():
    global latest_frame, car_count, bike_count

    cap = cv2.VideoCapture("samples/bikeincarout.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    print("Starting video processing...")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        frame = frame[150:800, :]

        vehicle_results = vehicle_model(frame, conf=0.5, iou=0.4, classes=[2, 3, 5, 7])
        detections = []

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

                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.line(frame, (0, midline), (frame_width, midline), (255, 0, 0), 2)
        cv2.putText(frame, f"Cars: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Bikes: {bike_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        latest_frame = buffer.tobytes()

def generate_video_stream():
    global latest_frame
    while True:
        if latest_frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')

@app.get("/")
def index():
    return Response(content="Vehicle Detection Dashboard is running!", media_type="text/html")

@app.get("/video_feed")
def video_feed():
    if latest_frame:
        return Response(content=latest_frame, media_type="image/jpeg")
    return Response(content="No frame available", media_type="text/plain")

@app.get("/get_counts")
def get_counts():
    total_capacity = 10
    filled_spaces = car_count + bike_count
    free_spaces = max(0, total_capacity - filled_spaces)

    return {
        'cars': car_count,
        'bikes': bike_count,
        'filled_spaces': filled_spaces,
        'free_spaces': free_spaces
    }

if __name__ == "__main__":
    threading.Thread(target=process_video, daemon=True).start()  # Run video processing in background
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, debug=True)
