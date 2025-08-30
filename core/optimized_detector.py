import torch
import numpy as np
from ultralytics import YOLO
import logging
from typing import List, Tuple, Optional
import cv2

logger = logging.getLogger(__name__)

class OptimizedVehicleDetector:
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._select_device(device)
        self.model_path = model_path
        self.model = self._load_optimized_model()
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.inference_times = []
        self.max_inference_history = 100
        
        logger.info(f"OptimizedVehicleDetector initialized on {self.device}")
    
    def _select_device(self, device: str) -> str:
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            else:
                device = 'cpu'
        return device
    
    def _load_optimized_model(self) -> YOLO:
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            model.model.eval()
            
            if self.device == 'cuda':
                model.model.half()
                
            logger.info(f"Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_vehicles(self, 
                       frame: np.ndarray, 
                       conf_threshold: float = 0.4,
                       iou_threshold: float = 0.5,
                       max_detections: int = 50) -> List[List[float]]:
        try:
            start_time = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
            end_time = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
            
            if start_time:
                start_time.record()
            
            results = self.model(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                classes=list(self.vehicle_classes.keys()),
                max_det=max_detections,
                verbose=False
            )
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                self._update_inference_times(inference_time)
            
            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)
                    
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        detections.append([x1, y1, x2, y2, conf[i], cls[i]])
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in vehicle detection: {e}")
            return []
    
    def _update_inference_times(self, inference_time: float):
        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.max_inference_history:
            self.inference_times.pop(0)
    
    def get_average_inference_time(self) -> float:
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)
    
    def get_performance_stats(self) -> dict:
        avg_time = self.get_average_inference_time()
        return {
            'average_inference_time_ms': avg_time,
            'fps': 1000.0 / avg_time if avg_time > 0 else 0,
            'total_inferences': len(self.inference_times),
            'device': self.device
        }
    
    def cleanup(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        logger.info("OptimizedVehicleDetector cleanup completed")
