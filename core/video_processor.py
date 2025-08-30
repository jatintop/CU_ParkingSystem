import cv2
import numpy as np
import threading
import time
import logging
from collections import deque

from sort.sort import Sort
from config.config import settings
from core.detector import VehicleDetector
from core.ocr import LicensePlateReader

logger = logging.getLogger(__name__)

class SmartParkingProcessor:
    def __init__(self):
        self.detector = VehicleDetector()
        self.reader = LicensePlateReader()
        self.tracker = Sort(
            max_age=settings.get('sort_max_age', 3),
            min_hits=settings.get('sort_min_hits', 3),
            iou_threshold=settings.get('sort_iou_threshold', 0.3)
        )

        self.car_count = 0
        self.bike_count = 0
        self.vehicle_directions = {}
        self.track_class_labels = {}
        self.tracked_vehicles = {}
        self.midline = settings.get('midline', 360)
        self.rtsp_link = settings.get('rtsp_link')
        self.video_path = settings.get('video_path') # For local video testing

        # Camera and frame processing variables
        self.cap = None
        self.connection_retry_count = 0
        self.max_retry_attempts = settings.get('max_retry_attempts', 5)
        self.base_retry_delay = settings.get('base_retry_delay', 2)
        self.camera_connected = False
        self.last_successful_frame_time = time.time()
        self.latest_frame_bytes = None
        self.lock = threading.Lock()

        self.processing_scale = settings.get('processing_scale', 0.5)
        self.target_width = settings.get('target_width', 1280)
        self.target_height = settings.get('target_height', 720)
        self.process_every_n_frames = settings.get('process_every_n_frames', 5)
        self.frame_skip_counter = 0
        self.frame_buffer = deque(maxlen=3) # For smoother playback

        self.class_names = {2: 'Car', 3: 'Bike', 5: 'Bus', 7: 'Truck'}

    def create_placeholder_frame(self, message="Connecting to camera..."):
        """Create a placeholder frame with a message"""
        placeholder = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        cv2.putText(placeholder, message, (50, self.target_height // 2 - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(placeholder, f"Status: {message}", (50, self.target_height // 2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        return placeholder

    def initialize_camera(self):
        """Initialize camera connection with robust error handling"""
        self.connection_retry_count = 0
        self.camera_connected = False
        retry_delay = self.base_retry_delay

        while self.connection_retry_count < self.max_retry_attempts:
            try:
                if self.rtsp_link:
                    logger.info(f"Attempting to connect to RTSP stream... (Attempt {self.connection_retry_count + 1})")
                else:
                    logger.info(f"Attempting to open local video file: {self.video_path}")

                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

                if self.rtsp_link:
                    connection_configs = [
                        {
                            'url': self.rtsp_link,
                            'backend': cv2.CAP_FFMPEG,
                            'props': {
                                cv2.CAP_PROP_BUFFERSIZE: 1,
                                cv2.CAP_PROP_FRAME_WIDTH: self.target_width,
                                cv2.CAP_PROP_FRAME_HEIGHT: self.target_height,
                                cv2.CAP_PROP_FPS: 15,
                                cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc('H', '2', '6', '4')
                            }
                        },
                        {
                            'url': self.rtsp_link.replace('?tcp', ''),
                            'backend': cv2.CAP_FFMPEG,
                            'props': {
                                cv2.CAP_PROP_BUFFERSIZE: 2,
                                cv2.CAP_PROP_FRAME_WIDTH: self.target_width,
                                cv2.CAP_PROP_FRAME_HEIGHT: self.target_height,
                            }
                        },
                        {
                            'url': self.rtsp_link,
                            'backend': cv2.CAP_ANY,
                            'props': {
                                cv2.CAP_PROP_BUFFERSIZE: 1,
                            }
                        }
                    ]
                else: # Use local video file
                    connection_configs = [
                        {
                            'url': self.video_path,
                            'backend': cv2.CAP_ANY,
                            'props': {} # No specific props for local file usually
                        }
                    ]

                for i, config in enumerate(connection_configs):
                    try:
                        logger.info(f"Trying connection method {i+1}: {config['url']}")
                        self.cap = cv2.VideoCapture(config['url'], config['backend'] if 'backend' in config else cv2.CAP_ANY)

                        if self.cap is None:
                            logger.warning(f"Failed to create VideoCapture object for method {i+1}")
                            continue

                        for prop, value in config['props'].items():
                            try:
                                self.cap.set(prop, value)
                            except Exception as prop_e:
                                logger.warning(f"Could not set property {prop}: {prop_e}")

                        if not self.cap.isOpened():
                            logger.warning(f"VideoCapture not opened for method {i+1}. Releasing and trying next method.")
                            if self.cap is not None:
                                self.cap.release()
                                self.cap = None
                            continue
                        else:
                            logger.info(f"VideoCapture successfully opened for method {i+1}.")

                        logger.info(f"Testing connection stability for method {i+1}...")
                        successful_reads = 0
                        test_frames = []

                        for test_attempt in range(8):
                            ret, test_frame = self.cap.read()
                            if ret and test_frame is not None and test_frame.size > 0:
                                frame_mean = np.mean(test_frame)
                                if frame_mean > 5:
                                    successful_reads += 1
                                    test_frames.append(test_frame)
                                    logger.debug(f"  Test read {test_attempt + 1}: SUCCESS (mean: {frame_mean:.1f})")
                                else:
                                    logger.warning(f"  Test read {test_attempt + 1}: Frame too dark (mean: {frame_mean:.1f})")
                            else:
                                logger.warning(f"  Test read {test_attempt + 1}: FAILED to read frame")
                            time.sleep(0.05)

                        if successful_reads >= 3 and len(test_frames) > 0:
                            self.camera_connected = True
                            logger.info(f"Camera connected successfully with method {i+1}!")
                            sample_frame = test_frames[-1]
                            if sample_frame.shape[1] != self.target_width or sample_frame.shape[0] != self.target_height:
                                sample_frame = cv2.resize(sample_frame, (self.target_width, self.target_height),
                                                        interpolation=cv2.INTER_LINEAR)
                            with self.lock:
                                _, buffer = cv2.imencode('.jpg', sample_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                                if buffer is not None:
                                    self.latest_frame_bytes = buffer.tobytes()
                            logger.info(f"initialize_camera returning True for method {i+1}")
                            return True
                        else:
                            logger.warning(f"Method {i+1} failed stability test: {successful_reads}/8 successful reads. Releasing and trying next method.")

                    except Exception as e:
                        logger.warning(f"Connection method {i+1} failed: {str(e)}. Releasing and trying next method.")
                    finally:
                        if self.cap is not None and not self.camera_connected:
                            self.cap.release()
                            self.cap = None

                logger.error("All connection methods failed for initialize_camera.")
                if self.rtsp_link:
                    logger.error("All connection methods failed for initialize_camera.")
                    raise Exception("All connection methods failed")
                else:
                    logger.error(f"Failed to open local video file: {self.video_path}")
                    return False # For local video, no retry loop here

            except Exception as e:
                logger.error(f"Camera/Video connection failed in initialize_camera: {str(e)}")
                self.connection_retry_count += 1
                self.camera_connected = False

                if self.connection_retry_count < self.max_retry_attempts and self.rtsp_link: # Only retry for RTSP
                    logger.info(f"Retrying initialize_camera in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 10)
                else:
                    logger.error("Max retry attempts reached or local video failed. Returning False.")
                    return False
        logger.info("initialize_camera returning False after loop.")
        return False

    def reconnect_camera(self):
        """Reconnect to camera when connection is lost"""
        logger.warning("Camera connection lost. Attempting to reconnect...")
        self.connection_retry_count = 0
        self.camera_connected = False
        return self.initialize_camera()

    def resize_frame_optimized(self, frame):
        """Optimized frame resizing - only resize once to target resolution"""
        if frame is None:
            return None

        height, width = frame.shape[:2]
        if width != self.target_width or height != self.target_height:
            frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
        return frame

    def process_single_frame(self, frame):
        """Processes a single frame for detection, tracking, and OCR, and draws overlays."""
        # Crop frame as per original logic
        frame = frame[150:800, :]
        frame_height, frame_width, _ = frame.shape

        detections = self.detector.detect_vehicles(frame)

        if detections:
            dets = np.array(detections)
            tracks = self.tracker.update(dets[:, :5])

            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                x1, y1, x2, y2 = map(int, bbox)

                # Ensure bounding box is within frame bounds
                x1 = max(0, min(x1, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                x2 = max(x1 + 1, min(x2, frame_width))
                y2 = max(y1 + 1, min(y2, frame_height))

                # Associate class label with track_id
                cls = None
                for det in detections:
                    det_x1, det_y1, det_x2, det_y2, det_conf, det_cls = det
                    if not (x2 < det_x1 or x1 > det_x2 or y2 < det_y1 or y1 > det_y2):
                        cls = det_cls
                        break

                if cls is not None:
                    self.track_class_labels[track_id] = cls

                # Detect crossing the midline
                center_y = (y1 + y2) // 2
                if track_id not in self.vehicle_directions:
                    self.vehicle_directions[track_id] = None

                if self.vehicle_directions[track_id] is None:
                    if center_y < self.midline:
                        self.vehicle_directions[track_id] = 'up'
                    else:
                        self.vehicle_directions[track_id] = 'down'
                else:
                    if self.vehicle_directions[track_id] == 'up' and center_y > self.midline:
                        if self.track_class_labels.get(track_id) == 2: # Car
                            self.car_count += 1
                            logger.info(f"Car entered: {self.car_count}")
                        elif self.track_class_labels.get(track_id) == 3: # Bike
                            self.bike_count += 1
                            logger.info(f"Bike entered: {self.bike_count}")
                        self.vehicle_directions[track_id] = 'crossed_down'
                    elif self.vehicle_directions[track_id] == 'down' and center_y < self.midline:
                        if self.track_class_labels.get(track_id) == 2: # Car
                            self.car_count = max(0, self.car_count - 1)
                            logger.info(f"Car exited: {self.car_count}")
                        elif self.track_class_labels.get(track_id) == 3: # Bike
                            self.bike_count = max(0, self.bike_count - 1)
                            logger.info(f"Bike exited: {self.bike_count}")
                        self.vehicle_directions[track_id] = 'crossed_up'

                # License plate detection and OCR
                if track_id not in self.tracked_vehicles and self.frame_skip_counter % 30 == 0: # Process less frequently
                    try:
                        vehicle_crop = frame[y1:y2, x1:x2]
                        if vehicle_crop.size != 0:
                            license_plate_results = self.detector.detect_license_plates(vehicle_crop)
                            for lp_result in license_plate_results:
                                lp_x1, lp_y1, lp_x2, lp_y2 = lp_result[:4]
                                # Adjust license plate coordinates to original frame
                                lp_x1_abs = lp_x1 + x1
                                lp_y1_abs = lp_y1 + y1
                                lp_x2_abs = lp_x2 + x1
                                lp_y2_abs = lp_y2 + y1

                                license_text = self.reader.read_license_plate(frame, [lp_x1_abs, lp_y1_abs, lp_x2_abs, lp_y2_abs])
                                self.tracked_vehicles[track_id] = license_text or 'N/A'
                                break
                    except Exception as e:
                        logger.error(f"License plate processing error: {str(e)}")
                        self.tracked_vehicles[track_id] = 'N/A'

                # Drawing bounding box and license plate text
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                class_name = self.class_names.get(self.track_class_labels.get(track_id, 0), 'Vehicle')

                cv2.putText(frame, f"ID:{track_id} {class_name}",
                           (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Plate: {self.tracked_vehicles.get(track_id, 'N/A')}",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Drawing midline and counts on frame
        cv2.line(frame, (0, self.midline), (frame_width, self.midline), (255, 0, 0), 3)
        cv2.putText(frame, f"Cars: {self.car_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Bikes: {self.bike_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Frame: {self.frame_skip_counter}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add connection status with better visibility
        status_color = (0, 255, 0) if self.camera_connected else (0, 0, 255)
        cv2.putText(frame, f"RTSP: {'Connected' if self.camera_connected else 'Disconnected'}",
                   (10, self.target_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return frame

    def start_processing_thread(self):
        """Starts a daemon thread to continuously process frames."""
        threading.Thread(target=self._process_frames_loop, daemon=True).start()

    def _process_frames_loop(self):
        """The main loop for reading and processing video frames."""
        if not self.initialize_camera():
            logger.error("Failed to initialize camera. Using placeholder frames.")

        consecutive_failures = 0
        max_consecutive_failures = 15
        last_log_time = time.time()

        while True:
            try:
                current_time = time.time()

                if self.cap is None or not self.cap.isOpened() or not self.camera_connected:
                    if self.rtsp_link:
                        logger.warning("Camera not connected. Attempting reconnection...")
                        if not self.reconnect_camera():
                            placeholder = self.create_placeholder_frame("Camera disconnected - Retrying...")
                            _, buffer = cv2.imencode('.jpg', placeholder)
                            if buffer is not None:
                                with self.lock:
                                    self.latest_frame_bytes = buffer.tobytes()
                            time.sleep(1)
                            continue
                    else: # Local video, no reconnection logic needed, just restart if it failed
                        logger.error(f"Local video file {self.video_path} is not open. Attempting to re-initialize.")
                        if not self.initialize_camera():
                            placeholder = self.create_placeholder_frame("Local video not found or corrupted.")
                            _, buffer = cv2.imencode('.jpg', placeholder)
                            if buffer is not None:
                                with self.lock:
                                    self.latest_frame_bytes = buffer.tobytes()
                            time.sleep(1)
                            continue

                ret, frame = self.cap.read()
                self.frame_skip_counter += 1
                logger.debug(f"Frame read attempt: ret={ret}, frame_size={frame.size if frame is not None else 'None'}")

                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if current_time - last_log_time > 5:
                        logger.warning(f"Frame read failed in _process_frames_loop (consecutive failures: {consecutive_failures}). {'Looping video.' if not self.rtsp_link else 'Attempting reconnection.'}")
                        last_log_time = current_time

                    if not self.rtsp_link: # If local video, loop it
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        logger.info(f"Local video {self.video_path} ended, looping.")
                        consecutive_failures = 0 # Reset failures for looping video
                        continue
                    elif consecutive_failures >= max_consecutive_failures: # Only for RTSP
                        logger.error("Too many consecutive frame failures in _process_frames_loop. Reconnecting camera...")
                        self.camera_connected = False
                        if not self.reconnect_camera():
                            time.sleep(2)
                        consecutive_failures = 0

                    placeholder = self.create_placeholder_frame("Reading frames - Please wait...")
                    _, buffer = cv2.imencode('.jpg', placeholder)
                    if buffer is not None:
                        with self.lock:
                            self.latest_frame_bytes = buffer.tobytes()
                        logger.debug("Updated latest_frame_bytes with placeholder due to read failure.")
                    time.sleep(0.05)
                    continue

                frame_mean = np.mean(frame)
                if frame_mean < 5:
                    consecutive_failures += 1
                    logger.warning(f"Frame appears invalid (mean pixel value: {frame_mean:.1f}). Consecutive failures: {consecutive_failures}")
                    continue

                consecutive_failures = 0
                self.last_successful_frame_time = current_time

                frame = self.resize_frame_optimized(frame)
                if frame is None:
                    logger.warning("Resized frame is None, skipping processing.")
                    continue

                self.frame_buffer.append(frame.copy())

                should_process_yolo = (self.frame_skip_counter % self.process_every_n_frames == 0)

                display_frame = frame.copy() # Frame to draw on

                if should_process_yolo and len(self.frame_buffer) > 0:
                    try:
                        processed_display_frame = self.process_single_frame(display_frame)
                        if processed_display_frame is not None:
                            display_frame = processed_display_frame
                        logger.debug("Frame processed by process_single_frame.")
                    except Exception as e:
                        logger.error(f"Error in YOLO processing within _process_frames_loop: {str(e)}")

                ret_encode, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ret_encode and buffer is not None:
                    with self.lock:
                        self.latest_frame_bytes = buffer.tobytes()
                    logger.debug("Updated latest_frame_bytes with processed frame.")
                else:
                    logger.error("Failed to encode processed frame to JPEG.")

            except Exception as e:
                logger.error(f"Unhandled error in _process_frames_loop: {str(e)}")
                consecutive_failures += 1
                time.sleep(0.02)

    def get_current_frame_bytes(self):
        """Returns the latest processed frame as bytes."""
        with self.lock:
            if self.latest_frame_bytes is None:
                placeholder = self.create_placeholder_frame("Initializing camera...")
                _, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 92])
                return buffer.tobytes() if buffer is not None else b''
            return self.latest_frame_bytes

    def get_counts(self):
        return {
            'cars': max(0, self.car_count),
            'bikes': max(0, self.bike_count),
            'filled_spaces': max(0, self.car_count + self.bike_count),
            'free_spaces': max(0, 10 - (self.car_count + self.bike_count)) # Assuming total capacity of 10
        }

    def reset_counts(self):
        self.car_count = 0
        self.bike_count = 0
        logger.info("Vehicle counts reset")

    def get_camera_status(self):
        is_connected = self.cap is not None and self.cap.isOpened() and self.camera_connected
        time_since_last_frame = time.time() - self.last_successful_frame_time

        return {
            "connected": is_connected,
            "resolution": f"{self.target_width}x{self.target_height}" if is_connected else "N/A",
            "last_frame_age": round(time_since_last_frame, 1),
            "status": "Active" if time_since_last_frame < 5 else "Stale"
        }

    def debug_rtsp_info(self):
        debug_info = {
            "opencv_version": cv2.__version__,
            "rtsp_url": self.rtsp_link,
            "target_resolution": f"{self.target_width}x{self.target_height}",
            "processing_scale": self.processing_scale,
            "process_every_n_frames": self.process_every_n_frames,
            "cap_backends": []
        }

        backends = [
            ("CAP_ANY", cv2.CAP_ANY),
            ("CAP_FFMPEG", cv2.CAP_FFMPEG),
        ]

        if hasattr(cv2, 'CAP_GSTREAMER'):
            backends.append(("CAP_GSTREAMER", cv2.CAP_GSTREAMER))
        if hasattr(cv2, 'CAP_DSHOW'):
            backends.append(("CAP_DSHOW", cv2.CAP_DSHOW))

        for name, backend in backends:
            try:
                test_cap = cv2.VideoCapture(self.rtsp_link, backend)
                is_opened = test_cap.isOpened() if test_cap else False
                debug_info["cap_backends"].append({
                    "name": name,
                    "value": backend,
                    "available": True,
                    "can_open": is_opened
                })
                if test_cap:
                    test_cap.release()
            except Exception as e:
                debug_info["cap_backends"].append({
                    "name": name,
                    "value": backend,
                    "available": False,
                    "error": str(e)
                })
        return debug_info

    def test_simple_rtsp_connection(self):
        try:
            test_cap = cv2.VideoCapture(self.rtsp_link)

            if not test_cap.isOpened():
                test_cap.release()
                return {
                    "success": False,
                    "error": "Could not open RTSP stream",
                    "url": self.rtsp_link
                }

            ret, frame = test_cap.read()
            test_cap.release()

            if ret and frame is not None:
                height, width = frame.shape[:2]
                mean_pixel = float(np.mean(frame))

                return {
                    "success": True,
                    "frame_shape": [height, width],
                    "mean_pixel_value": mean_pixel,
                    "url": self.rtsp_link
                }
            else:
                return {
                    "success": False,
                    "error": "Could not read frame from RTSP stream",
                    "url": self.rtsp_link
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": self.rtsp_link
            }
