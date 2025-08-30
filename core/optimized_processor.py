import cv2
import numpy as np
import threading
import time
import logging
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import psutil
import gc

from sort.sort import Sort
from config.config import settings
from core.optimized_detector import OptimizedVehicleDetector

logger = logging.getLogger(__name__)

class OptimizedParkingProcessor:
    def __init__(self):
        self.detector = OptimizedVehicleDetector(
            model_path=settings.get('vehicle_model_path'),
            device='auto'
        )
        
        self.tracker = Sort(
            max_age=settings.get('sort_max_age', 5),
            min_hits=settings.get('sort_min_hits', 2),
            iou_threshold=settings.get('sort_iou_threshold', 0.3)
        )
        
        self.car_count = 0
        self.bike_count = 0
        self.total_vehicles = 0
        
        self.vehicle_directions = {}
        self.track_class_labels = {}
        self.track_history = defaultdict(list)
        self.crossed_vehicles = set()
        
        self.midline = settings.get('midline', 360)
        self.rtsp_link = settings.get('rtsp_link')
        self.video_path = settings.get('video_path')
        
        self.target_width = settings.get('target_width', 1280)
        self.target_height = settings.get('target_height', 720)
        self.process_every_n_frames = settings.get('process_every_n_frames', 3)
        self.max_detections = settings.get('max_detections', 30)
        self.confidence_threshold = settings.get('confidence_threshold', 0.4)
        self.iou_threshold = settings.get('iou_threshold', 0.5)
        
        self.cap = None
        self.camera_connected = False
        self.connection_retry_count = 0
        self.max_retry_attempts = settings.get('max_retry_attempts', 3)
        self.retry_delay = settings.get('base_retry_delay', 1)
        
        self.latest_frame_bytes = None
        self.frame_skip_counter = 0
        self.last_successful_frame_time = time.time()
        self.processing_lock = threading.RLock()
        
        self.performance_stats = {
            'fps': 0,
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0,
            'total_frames_processed': 0,
            'detection_fps': 0
        }
        self.last_stats_update = time.time()
        
        self.processing_thread = None
        self.shutdown_event = threading.Event()
        
        logger.info("OptimizedParkingProcessor initialized")
    
    def initialize_camera(self) -> bool:
        """Initialize camera connection with improved error handling."""
        self.connection_retry_count = 0
        
        while self.connection_retry_count < self.max_retry_attempts:
            try:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                
                # Try different connection methods
                connection_methods = self._get_connection_methods()
                
                for method_name, method_config in connection_methods:
                    try:
                        logger.info(f"Trying connection method: {method_name}")
                        self.cap = cv2.VideoCapture(method_config['url'], method_config.get('backend', cv2.CAP_ANY))
                        
                        if self.cap is None or not self.cap.isOpened():
                            continue
                        
                        # Set camera properties
                        for prop, value in method_config.get('props', {}).items():
                            self.cap.set(prop, value)
                        
                        # Test connection stability
                        if self._test_connection_stability():
                            self.camera_connected = True
                            logger.info(f"Camera connected successfully using {method_name}")
                            return True
                        else:
                            self.cap.release()
                            self.cap = None
                            
                    except Exception as e:
                        logger.warning(f"Connection method {method_name} failed: {e}")
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                
                self.connection_retry_count += 1
                if self.connection_retry_count < self.max_retry_attempts:
                    logger.info(f"Retrying camera connection in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    self.retry_delay = min(self.retry_delay * 1.5, 5)
                    
            except Exception as e:
                logger.error(f"Camera initialization error: {e}")
                self.connection_retry_count += 1
        
        logger.error("Failed to initialize camera after all attempts")
        return False
    
    def _get_connection_methods(self) -> List[Tuple[str, Dict]]:
        """Get list of connection methods to try."""
        methods = []
        
        if self.rtsp_link:
            # RTSP connection methods
            methods.extend([
                ("RTSP_FFMPEG_OPTIMIZED", {
                    'url': self.rtsp_link,
                    'backend': cv2.CAP_FFMPEG,
                    'props': {
                        cv2.CAP_PROP_BUFFERSIZE: 1,
                        cv2.CAP_PROP_FRAME_WIDTH: self.target_width,
                        cv2.CAP_PROP_FRAME_HEIGHT: self.target_height,
                        cv2.CAP_PROP_FPS: 15
                    }
                }),
                ("RTSP_FFMPEG_SIMPLE", {
                    'url': self.rtsp_link,
                    'backend': cv2.CAP_FFMPEG,
                    'props': {
                        cv2.CAP_PROP_BUFFERSIZE: 2
                    }
                }),
                ("RTSP_ANY", {
                    'url': self.rtsp_link,
                    'backend': cv2.CAP_ANY,
                    'props': {}
                })
            ])
        else:
            # Local video file
            methods.append(("LOCAL_VIDEO", {
                'url': self.video_path,
                'backend': cv2.CAP_ANY,
                'props': {}
            }))
        
        return methods
    
    def _test_connection_stability(self) -> bool:
        """Test if camera connection is stable."""
        if not self.cap or not self.cap.isOpened():
            return False
        
        successful_reads = 0
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                if np.mean(frame) > 5:  # Check if frame is not too dark
                    successful_reads += 1
            time.sleep(0.1)
        
        return successful_reads >= 3
    
    def start_processing_thread(self):
        """Start the background processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.shutdown_event.clear()
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="VideoProcessor"
            )
            self.processing_thread.start()
            logger.info("Processing thread started")
    
    def _processing_loop(self):
        """Main processing loop for video frames."""
        if not self.initialize_camera():
            logger.error("Failed to initialize camera")
            return
        
        consecutive_failures = 0
        max_consecutive_failures = 10
        frame_times = deque(maxlen=30)  # Track FPS
        
        while not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Check camera connection
                if not self._is_camera_healthy():
                    if not self._handle_camera_failure():
                        time.sleep(1)
                        continue
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning("Too many consecutive frame failures")
                        if not self._handle_camera_failure():
                            break
                    time.sleep(0.05)
                    continue
                
                consecutive_failures = 0
                self.frame_skip_counter += 1
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Update frame buffer
                if processed_frame is not None:
                    with self.processing_lock:
                        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        self.latest_frame_bytes = buffer.tobytes()
                
                # Update performance stats
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                self._update_performance_stats(frame_times)
                
                # Adaptive frame rate control
                target_fps = 30
                target_frame_time = 1.0 / target_fps
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
    
    def _is_camera_healthy(self) -> bool:
        """Check if camera connection is healthy."""
        return (self.cap is not None and 
                self.cap.isOpened() and 
                self.camera_connected and
                time.time() - self.last_successful_frame_time < 10)
    
    def _handle_camera_failure(self) -> bool:
        """Handle camera connection failure."""
        logger.warning("Camera connection failure detected")
        self.camera_connected = False
        
        if self.rtsp_link:
            # Try to reconnect for RTSP
            return self.initialize_camera()
        else:
            # For local video, try to restart
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True
    
    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a single frame for vehicle detection and counting."""
        try:
            # Resize frame for processing
            frame = self._resize_frame(frame)
            if frame is None:
                return None
            
            # Crop frame (remove top portion as in original)
            frame = frame[150:800, :]
            frame_height, frame_width = frame.shape[:2]
            
            # Run detection only every N frames
            if self.frame_skip_counter % self.process_every_n_frames == 0:
                detections = self.detector.detect_vehicles(
                    frame,
                    conf_threshold=self.confidence_threshold,
                    iou_threshold=self.iou_threshold,
                    max_detections=self.max_detections
                )
                
                if detections:
                    self._update_tracking(frame, detections, frame_width, frame_height)
            
            # Draw overlays
            display_frame = self._draw_overlays(frame, frame_width, frame_height)
            
            self.last_successful_frame_time = time.time()
            self.performance_stats['total_frames_processed'] += 1
            
            return display_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None
    
    def _resize_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Resize frame to target resolution."""
        if frame is None:
            return None
        
        height, width = frame.shape[:2]
        if width != self.target_width or height != self.target_height:
            frame = cv2.resize(frame, (self.target_width, self.target_height), 
                             interpolation=cv2.INTER_LINEAR)
        return frame
    
    def _update_tracking(self, frame: np.ndarray, detections: List[List[float]], 
                        frame_width: int, frame_height: int):
        """Update vehicle tracking and counting."""
        try:
            # Convert detections to numpy array
            dets = np.array(detections)
            
            # Update tracker
            tracks = self.tracker.update(dets[:, :5])
            
            # Process each track
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Ensure bounding box is within frame bounds
                x1 = max(0, min(x1, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                x2 = max(x1 + 1, min(x2, frame_width))
                y2 = max(y1 + 1, min(y2, frame_height))
                
                # Get vehicle class
                cls = self._get_vehicle_class(detections, bbox)
                if cls is not None:
                    self.track_class_labels[track_id] = cls
                
                # Update vehicle counting
                self._update_vehicle_count(track_id, x1, y1, x2, y2)
                
                # Update track history
                center_y = (y1 + y2) // 2
                self.track_history[track_id].append(center_y)
                if len(self.track_history[track_id]) > 10:
                    self.track_history[track_id].pop(0)
                    
        except Exception as e:
            logger.error(f"Error updating tracking: {e}")
    
    def _get_vehicle_class(self, detections: List[List[float]], bbox: np.ndarray) -> Optional[int]:
        """Get vehicle class for a bounding box."""
        x1, y1, x2, y2 = bbox
        
        for det in detections:
            det_x1, det_y1, det_x2, det_y2, _, det_cls = det
            # Check if bounding boxes overlap
            if not (x2 < det_x1 or x1 > det_x2 or y2 < det_y1 or y1 > det_y2):
                return int(det_cls)
        return None
    
    def _update_vehicle_count(self, track_id: int, x1: int, y1: int, x2: int, y2: int):
        """Update vehicle count based on midline crossing."""
        center_y = (y1 + y2) // 2
        
        # Initialize direction if not set
        if track_id not in self.vehicle_directions:
            self.vehicle_directions[track_id] = 'up' if center_y < self.midline else 'down'
            return
        
        current_direction = self.vehicle_directions[track_id]
        vehicle_class = self.track_class_labels.get(track_id)
        
        # Check for midline crossing
        if current_direction == 'up' and center_y > self.midline:
            # Vehicle crossed from top to bottom (entering)
            if track_id not in self.crossed_vehicles:
                if vehicle_class == 2:  # Car
                    self.car_count += 1
                elif vehicle_class == 3:  # Motorcycle
                    self.bike_count += 1
                self.crossed_vehicles.add(track_id)
                logger.info(f"Vehicle {track_id} entered - Cars: {self.car_count}, Bikes: {self.bike_count}")
            
        elif current_direction == 'down' and center_y < self.midline:
            # Vehicle crossed from bottom to top (exiting)
            if track_id not in self.crossed_vehicles:
                if vehicle_class == 2:  # Car
                    self.car_count = max(0, self.car_count - 1)
                elif vehicle_class == 3:  # Motorcycle
                    self.bike_count = max(0, self.bike_count - 1)
                self.crossed_vehicles.add(track_id)
                logger.info(f"Vehicle {track_id} exited - Cars: {self.car_count}, Bikes: {self.bike_count}")
        
        # Reset direction if vehicle has moved significantly
        if abs(center_y - self.midline) > 50:
            self.vehicle_directions[track_id] = 'up' if center_y < self.midline else 'down'
            self.crossed_vehicles.discard(track_id)
    
    def _draw_overlays(self, frame: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
        """Draw overlays on frame."""
        # Draw midline
        cv2.line(frame, (0, self.midline), (frame_width, self.midline), (255, 0, 0), 2)
        
        # Draw counts
        cv2.putText(frame, f"Cars: {self.car_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Bikes: {self.bike_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw performance stats
        fps_text = f"FPS: {self.performance_stats['fps']:.1f}"
        cv2.putText(frame, fps_text, (10, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw connection status
        status_color = (0, 255, 0) if self.camera_connected else (0, 0, 255)
        status_text = "Connected" if self.camera_connected else "Disconnected"
        cv2.putText(frame, status_text, (frame_width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return frame
    
    def _update_performance_stats(self, frame_times: deque):
        """Update performance statistics."""
        current_time = time.time()
        if current_time - self.last_stats_update > 1.0:  # Update every second
            # Calculate FPS
            if len(frame_times) > 0:
                avg_frame_time = sum(frame_times) / len(frame_times)
                self.performance_stats['fps'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Get system stats
            process = psutil.Process()
            self.performance_stats['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            self.performance_stats['cpu_usage_percent'] = process.cpu_percent()
            
            # Get detection stats
            detector_stats = self.detector.get_performance_stats()
            self.performance_stats['detection_fps'] = detector_stats.get('fps', 0)
            
            self.last_stats_update = current_time
    
    def get_current_frame_bytes(self) -> bytes:
        """Get current frame as bytes."""
        with self.processing_lock:
            if self.latest_frame_bytes is None:
                # Create placeholder frame
                placeholder = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Initializing...", (50, self.target_height // 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', placeholder)
                return buffer.tobytes() if buffer is not None else b''
            return self.latest_frame_bytes
    
    def get_counts(self) -> Dict[str, int]:
        """Get current vehicle counts."""
        return {
            'cars': max(0, self.car_count),
            'bikes': max(0, self.bike_count),
            'total': max(0, self.car_count + self.bike_count)
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_counts(self):
        """Reset vehicle counts."""
        self.car_count = 0
        self.bike_count = 0
        self.crossed_vehicles.clear()
        logger.info("Vehicle counts reset")
    
    def cleanup(self):
        """Clean up resources."""
        self.shutdown_event.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.detector.cleanup()
        
        # Clear memory
        self.track_history.clear()
        self.vehicle_directions.clear()
        self.track_class_labels.clear()
        self.crossed_vehicles.clear()
        
        gc.collect()
        
        logger.info("OptimizedParkingProcessor cleanup completed")
