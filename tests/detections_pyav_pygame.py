import av
import pygame
import torch
import numpy as np
import easyocr
from ultralytics import YOLO
from sort.sort import Sort

# Initialize Pygame
pygame.init()

# Load YOLO models and tracker
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vehicle_model = YOLO('artifacts/yolo11n.pt').to(device)
license_plate_model = YOLO('artifacts/license_plate_detector.pt').to(device)
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.4)
reader = easyocr.Reader(['en'])

# Video source
video_path = "samples/bikeincarout.mp4"
container = av.open(video_path)
video_stream = next(s for s in container.streams if s.type == 'video')

# Get video properties
frame_width = video_stream.codec_context.width
frame_height = video_stream.codec_context.height
midline = frame_height // 2

# Create Pygame window
screen = pygame.display.set_mode((frame_width, frame_height))
pygame.display.set_caption("Vehicle Detection - PyAV + Pygame")

# Variables
car_count, bike_count = 0, 0
vehicle_directions, track_class_labels, tracked_vehicles = {}, {}, {}

# Cropping region
CROP_Y1, CROP_Y2 = 150, 800


def preprocess_license_plate(license_plate):
    """Preprocess license plate for OCR."""
    return np.mean(license_plate, axis=2).astype(np.uint8)  # Convert to grayscale


def read_license_plate(frame, bbox):
    """Read license plate text using EasyOCR."""
    x1, y1, x2, y2 = map(int, bbox)
    license_plate = frame[y1:y2, x1:x2]
    if license_plate.size == 0:
        return None

    gray_plate = preprocess_license_plate(license_plate)
    results = reader.readtext(gray_plate)
    text = ' '.join([result[1] for result in results])
    return text.strip() if text else None


def detect_vehicles(frame):
    """Detect vehicles using YOLO."""
    results = vehicle_model(frame, conf=0.4, iou=0.3, classes=[2, 3, 5, 7])
    detections = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
        conf = result.conf[0].cpu().numpy()
        cls = int(result.cls[0].cpu().numpy())
        detections.append([x1, y1, x2, y2, conf, cls])
    return detections


def update_tracker(detections):
    """Update SORT tracker."""
    return tracker.update(np.array(detections)[:, :5])


def detect_license_plate(frame, track_id, bbox):
    """Detect and recognize license plates."""
    if track_id not in tracked_vehicles:
        x1, y1, x2, y2 = map(int, bbox)
        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            return

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


def draw_annotations(surface, track_id, bbox, color):
    """Draw bounding boxes and text in Pygame."""
    x1, y1, x2, y2 = map(int, bbox)
    pygame.draw.rect(surface, color, (x1, y1, x2 - x1, y2 - y1), 2)
    font = pygame.font.Font(None, 30)
    text = f"ID: {track_id} Plate: {tracked_vehicles.get(track_id, 'N/A')}"
    text_surface = font.render(text, True, (255, 255, 255))
    surface.blit(text_surface, (x1, y1 - 20))


# Main loop
running = True
frame_iter = iter(container.decode(video_stream))

while running:
    try:
        frame = next(frame_iter).to_ndarray(format="rgb24")
    except StopIteration:
        break  # Stop when video ends

    frame = frame[CROP_Y1:CROP_Y2, :]
    detections = detect_vehicles(frame)

    tracks = update_tracker(detections) if detections else []
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))  # Convert frame to Pygame surface

    for track in tracks:
        track_id = int(track[4])
        bbox = track[:4]

        detect_license_plate(frame, track_id, bbox)
        draw_annotations(surface, track_id, bbox, (0, 255, 0))

    # Draw midline and counts
    pygame.draw.line(surface, (255, 0, 0), (0, midline), (frame_width, midline), 2)
    font = pygame.font.Font(None, 40)
    car_text = font.render(f"Cars: {car_count}", True, (0, 255, 0))
    bike_text = font.render(f"Bikes: {bike_count}", True, (0, 255, 0))
    surface.blit(car_text, (10, 10))
    surface.blit(bike_text, (10, 50))

    # Display frame in Pygame
    screen.blit(surface, (0, 0))
    pygame.display.flip()


pygame.quit()
