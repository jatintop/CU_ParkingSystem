import torch
import av
import io
import threading
import numpy as np
from PIL import Image
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from sort.sort import Sort
import easyocr

app = FastAPI()

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vehicle_model = YOLO('artifacts/yolo11n.pt').to(device)
license_plate_model = YOLO('artifacts/license_plate_detector.pt').to(device)
tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)
reader = easyocr.Reader(['en'])

video_path = "samples/bikeincarout.mp4"

# Tracking variables
car_count = 0
bike_count = 0
vehicle_directions = {}
track_class_labels = {}
tracked_vehicles = {}
frame_buffer = None
overlay_data = []
lock = threading.Lock()


def read_license_plate(frame, bbox):
    """Extracts text from detected license plate region."""
    x1, y1, x2, y2 = map(int, bbox)
    license_plate = frame[y1:y2, x1:x2]
    gray_plate = np.mean(license_plate, axis=-1)  # Convert to grayscale
    results = reader.readtext(gray_plate)
    text = ' '.join([result[1] for result in results])
    return text.strip() if text else None


def process_video():
    """Continuously processes video and updates frame buffer."""
    global frame_buffer, overlay_data
    container = av.open(video_path)
    stream = next(s for s in container.streams if s.type == 'video')

    for frame in container.decode(stream):
        img = np.array(frame.to_image())  # Convert frame to NumPy array

        # Object detection
        vehicle_results = vehicle_model(img, conf=0.5, iou=0.4, classes=[2, 3, 5, 7])
        detections = []
        for result in vehicle_results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            conf = result.conf[0].item()
            cls = int(result.cls[0].item())
            detections.append([x1, y1, x2, y2, conf, cls])

        dets = np.array(detections) if detections else np.empty((0, 6))

        # Tracking
        tracks = tracker.update(dets[:, :5]) if len(dets) > 0 else []

        midline = img.shape[0] // 2
        frame_width = img.shape[1]
        overlay_data = []  # Clear overlay text data

        for track in tracks:
            track_id = int(track[4])
            x1, y1, x2, y2 = map(int, track[:4])

            cls = next((det[5] for det in detections if x1 <= det[0] <= x2 and y1 <= det[1] <= y2), None)
            if cls is not None:
                track_class_labels[track_id] = cls

            # Vehicle counting logic
            center_y = (y1 + y2) // 2
            if track_id not in vehicle_directions:
                vehicle_directions[track_id] = None
            if vehicle_directions[track_id] is None:
                vehicle_directions[track_id] = 'up' if center_y < midline else 'down'
            elif vehicle_directions[track_id] == 'up' and center_y > midline:
                if track_class_labels.get(track_id) == 2:
                    global car_count
                    car_count += 1
                elif track_class_labels.get(track_id) == 3:
                    global bike_count
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
                vehicle_crop = img[y1:y2, x1:x2]
                if vehicle_crop.size > 0:
                    license_plate_results = license_plate_model(vehicle_crop, conf=0.5, iou=0.4)
                    for lp_result in license_plate_results[0].boxes:
                        lp_x1, lp_y1, lp_x2, lp_y2 = lp_result.xyxy[0].tolist()
                        lp_x1 += x1
                        lp_y1 += y1
                        lp_x2 += x1
                        lp_y2 += y1
                        license_text = read_license_plate(img, [lp_x1, lp_y1, lp_x2, lp_y2])
                        tracked_vehicles[track_id] = license_text
                        break

            # Store overlay data for HTML
            overlay_data.append({"id": track_id, "x": x1, "y": y1, "plate": tracked_vehicles.get(track_id, "N/A")})

        # Store processed frame
        with lock:
            frame_buffer = img


def generate_frames():
    """Yields processed frames as a video stream."""
    global frame_buffer
    while True:
        with lock:
            if frame_buffer is not None:
                img = Image.fromarray(frame_buffer)
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="JPEG")
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + img_bytes.getvalue() + b'\r\n'


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/overlay_data")
def get_overlay_data():
    """Returns JSON overlay data (vehicle IDs, positions, license plates)."""
    return {"cars": car_count, "bikes": bike_count, "vehicles": overlay_data}


@app.get("/")
def index():
    return Response(content=f"""
    <html>
        <head>
            <script>
                async function updateOverlay() {{
                    let response = await fetch("/overlay_data");
                    let data = await response.json();
                    document.getElementById("car_count").innerText = "Cars: " + data.cars;
                    document.getElementById("bike_count").innerText = "Bikes: " + data.bikes;
                    
                    let overlay = document.getElementById("overlay");
                    overlay.innerHTML = "";
                    data.vehicles.forEach(v => {{
                        let div = document.createElement("div");
                        div.style.position = "absolute";
                        div.style.left = (v.x + 10) + "px";
                        div.style.top = (v.y + 10) + "px";
                        div.style.color = "red";
                        div.style.fontSize = "14px";
                        div.innerText = "ID: " + v.id + " Plate: " + v.plate;
                        overlay.appendChild(div);
                    }});
                }}

                setInterval(updateOverlay, 500);
            </script>
        </head>
        <body>
            <h1>Live Video Feed</h1>
            <p id="car_count">Cars: 0</p>
            <p id="bike_count">Bikes: 0</p>
            <div style="position: relative;">
                <img src="/video_feed" width="640" height="480" />
                <div id="overlay" style="position: absolute; top: 0; left: 0;"></div>
            </div>
        </body>
    </html>
    """, media_type="text/html")


# Start video processing in background
threading.Thread(target=process_video, daemon=True).start()
