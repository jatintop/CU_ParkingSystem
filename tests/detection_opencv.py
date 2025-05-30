import cv2
import torch
import easyocr
import numpy as np
from ultralytics import YOLO

from sort.sort import Sort

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")

vehicle_model = YOLO('artifacts/yolo11n.pt').to(device) 
license_plate_model = YOLO('artifacts/license_plate_detector.pt').to(device)
tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)

reader = easyocr.Reader(['en'])

video_path = "samples/bikeincarout.mp4"
cap = cv2.VideoCapture(video_path)

car_count = 0
bike_count = 0
vehicle_directions = {}
track_class_labels = {}
tracked_vehicles = {}

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
midline = 360

def read_license_plate(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    license_plate = frame[y1:y2, x1:x2]
    gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray_plate)
    text = ' '.join([result[1] for result in results])
    if text != '':
        return text.strip()
    return None


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = frame[150:800, :]

    vehicle_results = vehicle_model(frame, conf=0.5, iou=0.4, classes=[2, 3, 5, 7])
    detections = []
    for result in vehicle_results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].clone()
        conf = result.conf[0]
        cls = int(result.cls[0])
        detections.append([x1, y1, x2, y2, conf, cls])

    if detections != []:
        dets = np.array(detections)

        tracks = tracker.update(dets[:, :5])

        for track in tracks:
            track_id = int(track[4])
            bbox = track[:4]
            x1, y1, x2, y2 = map(int, bbox)

            # Find the corresponding class label for the track ID   
            cls = None
            for det in detections:
                det_x1, det_y1, det_x2, det_y2, det_conf, det_cls = det
                if (x1 <= det_x1 <= x2 and y1 <= det_y1 <= y2) or (x1 <= det_x2 <= x2 and y1 <= det_y2 <= y2):
                    cls = det_cls
                    break

            if cls is not None:
                track_class_labels[track_id] = cls

            # Calculate the center of the bounding box
            center_y = (y1 + y2) // 2

            # Check if the vehicle crosses the midline
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

            # Detecting license plate only if not already recognized
            if track_id not in tracked_vehicles:
                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size == 0:
                    continue

                license_plate_results = license_plate_model(vehicle_crop, conf=0.5, iou=0.4)
                for lp_result in license_plate_results[0].boxes:
                    lp_x1, lp_y1, lp_x2, lp_y2 = lp_result.xyxy[0].clone()
                    lp_conf = lp_result.conf[0]

                    # Adjusting the license plate bbox to the original frame coordinates
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
    cv2.putText(frame, f"Cars: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Bikes: {bike_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(f'Parking Lot System (running on {device})', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total Cars Detected: {car_count}")
print(f"Total Bikes Detected: {bike_count}")
print("Vehicle License Plates:")
for vehicle_id, plate in tracked_vehicles.items():
    print(f"ID {vehicle_id}: {plate if plate else 'N/A'}")
