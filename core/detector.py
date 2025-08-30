import torch
from ultralytics import YOLO
from config.config import settings

class VehicleDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"VehicleDetector running on {self.device}")
        self.vehicle_model = YOLO(settings.get('vehicle_model_path')).to(self.device)
        self.license_plate_model = YOLO(settings.get('license_plate_model_path')).to(self.device)
        self.vehicle_classes = settings.get('vehicle_classes', [])

    def detect_vehicles(self, frame, conf=0.5, iou=0.4):
        results = self.vehicle_model(frame, conf=conf, iou=iou, classes=self.vehicle_classes)
        detections = []
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
            conf = result.conf[0].cpu().item()
            cls = int(result.cls[0].cpu().item())
            detections.append([x1, y1, x2, y2, conf, cls])
        return detections

    def detect_license_plates(self, vehicle_crop, conf=0.5, iou=0.4):
        results = self.license_plate_model(vehicle_crop, conf=conf, iou=iou)
        license_plate_detections = []
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
            conf = result.conf[0].cpu().item()
            cls = int(result.cls[0].cpu().item())
            license_plate_detections.append([x1, y1, x2, y2, conf, cls])
        return license_plate_detections
