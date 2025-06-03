from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
import time
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vehicle_model = YOLO('artifacts/yolo11n.pt').to(device)
license_plate_model = YOLO('artifacts/license_plate_detector.pt').to(device)

tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)
reader = easyocr.Reader(['en'])

rtsp_link = "rtsp://smartparking:parking@2025@10.4.10.18?tcp"

cap = None
connection_retry_count = 0
max_retry_attempts = 5
base_retry_delay = 2

# Shared Data
latest_frame = None
lock = threading.Lock()
car_count = 0
bike_count = 0
vehicle_directions = {}
track_class_labels = {}
tracked_vehicles = {}
frame_width = 1280
frame_height = 720
midline = 360
camera_connected = False
last_successful_frame_time = time.time()

# Performance optimization variables
PROCESSING_SCALE = 0.5  # Scale down for processing, display original
TARGET_WIDTH = 1280  # Target display width
TARGET_HEIGHT = 720  # Target display height
PROCESS_EVERY_N_FRAMES = 5  # Process YOLO every 5th frame instead of every 2nd
frame_skip_counter = 0

def create_placeholder_frame(message="Connecting to camera..."):
    """Create a placeholder frame with a message"""
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, message, (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(placeholder, f"Status: {message}", (50, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
    return placeholder

def initialize_camera():
    """Initialize camera connection with robust error handling"""
    global cap, frame_width, frame_height, midline, connection_retry_count, camera_connected
    
    retry_delay = base_retry_delay
    
    while connection_retry_count < max_retry_attempts:
        try:
            logger.info(f"Attempting to connect to RTSP stream... (Attempt {connection_retry_count + 1})")
            
            # Release existing connection
            if cap is not None:
                cap.release()
                cap = None
            
            # Optimized connection configs prioritizing stable playback
            connection_configs = [
                # Method 1: TCP with optimized buffer settings
                {
                    'url': rtsp_link,
                    'backend': cv2.CAP_FFMPEG,
                    'props': {
                        cv2.CAP_PROP_BUFFERSIZE: 1,  # Minimize buffering
                        cv2.CAP_PROP_FRAME_WIDTH: TARGET_WIDTH,
                        cv2.CAP_PROP_FRAME_HEIGHT: TARGET_HEIGHT,
                        cv2.CAP_PROP_FPS: 15,
                        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc('H', '2', '6', '4')
                    }
                },
                # Method 2: Different codec approach
                {
                    'url': rtsp_link.replace('?tcp', ''),
                    'backend': cv2.CAP_FFMPEG,
                    'props': {
                        cv2.CAP_PROP_BUFFERSIZE: 2,
                        cv2.CAP_PROP_FRAME_WIDTH: TARGET_WIDTH,
                        cv2.CAP_PROP_FRAME_HEIGHT: TARGET_HEIGHT,
                    }
                },
                # Method 3: Any backend fallback
                {
                    'url': rtsp_link,
                    'backend': cv2.CAP_ANY,
                    'props': {
                        cv2.CAP_PROP_BUFFERSIZE: 1,
                    }
                }
            ]
            
            for i, config in enumerate(connection_configs):
                try:
                    logger.info(f"Trying connection method {i+1}: {config['url']}")
                    
                    # Create capture object
                    cap = cv2.VideoCapture(config['url'], config['backend'])
                    
                    if cap is None:
                        logger.warning(f"Failed to create VideoCapture object for method {i+1}")
                        continue
                    
                    # Configure capture properties
                    for prop, value in config['props'].items():
                        try:
                            cap.set(prop, value)
                        except Exception as prop_e:
                            logger.warning(f"Could not set property {prop}: {prop_e}")
                    
                    # Check if capture is opened
                    if not cap.isOpened():
                        logger.warning(f"VideoCapture not opened for method {i+1}")
                        if cap is not None:
                            cap.release()
                            cap = None
                        continue
                    
                    # Test the connection with stability check
                    logger.info(f"Testing connection stability for method {i+1}...")
                    successful_reads = 0
                    test_frames = []
                    
                    for test_attempt in range(8):  # Reduced test frames
                        try:
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None and test_frame.size > 0:
                                frame_mean = np.mean(test_frame)
                                if frame_mean > 5:
                                    successful_reads += 1
                                    test_frames.append(test_frame)
                                    logger.info(f"  Test read {test_attempt + 1}: SUCCESS (mean: {frame_mean:.1f})")
                                else:
                                    logger.warning(f"  Test read {test_attempt + 1}: Frame too dark (mean: {frame_mean:.1f})")
                            else:
                                logger.warning(f"  Test read {test_attempt + 1}: FAILED to read frame")
                        except Exception as read_e:
                            logger.warning(f"  Test read {test_attempt + 1}: Exception - {read_e}")
                        
                        time.sleep(0.05)  # Shorter delay
                    
                    if successful_reads >= 3 and len(test_frames) > 0:
                        # Use the target resolution, don't get actual from frame
                        frame_width = TARGET_WIDTH
                        frame_height = TARGET_HEIGHT
                        midline = frame_height // 2
                        
                        logger.info(f"Camera connected successfully with method {i+1}!")
                        logger.info(f"Target resolution: {frame_width}x{frame_height}")
                        logger.info(f"Successful reads: {successful_reads}/8")
                        
                        connection_retry_count = 0
                        camera_connected = True
                        
                        # Store a resized frame immediately
                        sample_frame = test_frames[-1]
                        # Resize to target resolution if needed
                        if sample_frame.shape[1] != TARGET_WIDTH or sample_frame.shape[0] != TARGET_HEIGHT:
                            sample_frame = cv2.resize(sample_frame, (TARGET_WIDTH, TARGET_HEIGHT), 
                                                    interpolation=cv2.INTER_LINEAR)
                        
                        with lock:
                            _, buffer = cv2.imencode('.jpg', sample_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                            if buffer is not None:
                                global latest_frame
                                latest_frame = buffer.tobytes()
                        
                        return True
                    else:
                        logger.warning(f"Method {i+1} failed stability test: {successful_reads}/8 successful reads")
                
                except Exception as e:
                    logger.warning(f"Connection method {i+1} failed: {str(e)}")
                    
                finally:
                    if cap is not None and not camera_connected:
                        cap.release()
                        cap = None
            
            raise Exception("All connection methods failed")
                
        except Exception as e:
            logger.error(f"Camera connection failed: {str(e)}")
            connection_retry_count += 1
            camera_connected = False
            
            if connection_retry_count < max_retry_attempts:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 10)
            else:
                logger.error("Max retry attempts reached. Camera connection failed.")
                return False
    
    return False

def reconnect_camera():
    """Reconnect to camera when connection is lost"""
    global connection_retry_count, camera_connected
    logger.warning("Camera connection lost. Attempting to reconnect...")
    connection_retry_count = 0
    camera_connected = False
    return initialize_camera()

def resize_frame_optimized(frame):
    """Optimized frame resizing - only resize once to target resolution"""
    if frame is None:
        return None
        
    height, width = frame.shape[:2]
    
    # Only resize if frame is not already target resolution
    if width != TARGET_WIDTH or height != TARGET_HEIGHT:
        # Log only on significant size changes to reduce spam
        if abs(width - TARGET_WIDTH) > 100 or abs(height - TARGET_HEIGHT) > 100:
            logger.info(f"Resizing frame from {width}x{height} to {TARGET_WIDTH}x{TARGET_HEIGHT}")
        
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    return frame

def read_license_plate(frame, bbox):
    """Reads license plate using OCR"""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            return None
            
        license_plate = frame[y1:y2, x1:x2]
        if license_plate.size == 0:
            return None
            
        gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray_plate)
        text = ' '.join([result[1] for result in results])
        return text.strip() if text else None
    except Exception as e:
        logger.error(f"Error reading license plate: {str(e)}")
        return None

def process_frames():
    """Optimized frame processing with reduced computational overhead"""
    global latest_frame, car_count, bike_count, cap, frame_width, frame_height
    global midline, camera_connected, last_successful_frame_time, frame_skip_counter
    
    # Initialize camera connection
    if not initialize_camera():
        logger.error("Failed to initialize camera. Using placeholder frames.")
    
    consecutive_failures = 0
    max_consecutive_failures = 15
    frame_count = 0
    last_log_time = time.time()
    
    # Frame buffer for smoother playback
    frame_buffer = deque(maxlen=3)
    
    while True:
        try:
            current_time = time.time()
            
            # Check if camera is connected and working
            if cap is None or not cap.isOpened() or not camera_connected:
                logger.warning("Camera not connected. Attempting reconnection...")
                if not reconnect_camera():
                    placeholder = create_placeholder_frame("Camera disconnected - Retrying...")
                    _, buffer = cv2.imencode('.jpg', placeholder)
                    if buffer is not None:
                        with lock:
                            latest_frame = buffer.tobytes()
                    time.sleep(1)
                    continue
            
            # Try to read frame
            ret, frame = cap.read()
            frame_count += 1
            frame_skip_counter += 1
            
            # Validate frame
            if not ret or frame is None or frame.size == 0:
                consecutive_failures += 1
                
                if current_time - last_log_time > 5:
                    logger.warning(f"Frame read failed (consecutive failures: {consecutive_failures})")
                    last_log_time = current_time
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many consecutive frame failures. Reconnecting camera...")
                    camera_connected = False
                    if not reconnect_camera():
                        time.sleep(2)
                    consecutive_failures = 0
                
                placeholder = create_placeholder_frame("Reading frames - Please wait...")
                _, buffer = cv2.imencode('.jpg', placeholder)
                if buffer is not None:
                    with lock:
                        latest_frame = buffer.tobytes()
                
                time.sleep(0.05)  # Shorter sleep
                continue
            
            # Check frame validity
            frame_mean = np.mean(frame)
            if frame_mean < 5:
                consecutive_failures += 1
                logger.warning(f"Frame appears invalid (mean pixel value: {frame_mean:.1f})")
                continue
            
            # Reset failure counter and update last successful frame time
            consecutive_failures = 0
            last_successful_frame_time = current_time
            
            # Resize frame to target resolution (do this once)
            frame = resize_frame_optimized(frame)
            if frame is None:
                continue
            
            # Add frame to buffer for smoother playback
            frame_buffer.append(frame.copy())
            
            # Process with YOLO less frequently to reduce computational load
            should_process_yolo = (frame_skip_counter % PROCESS_EVERY_N_FRAMES == 0)
            
            if should_process_yolo and len(frame_buffer) > 0:
                # Use a smaller frame for processing to speed up inference
                process_frame = cv2.resize(frame, 
                                         (int(TARGET_WIDTH * PROCESSING_SCALE), 
                                          int(TARGET_HEIGHT * PROCESSING_SCALE)), 
                                         interpolation=cv2.INTER_LINEAR)
                
                try:
                    # Vehicle detection and tracking
                    vehicle_results = vehicle_model(process_frame, conf=0.35, iou=0.45, classes=[2, 3, 5, 7])
                    detections = []

                    # Extract detections and scale back to original size
                    if hasattr(vehicle_results[0], 'boxes') and vehicle_results[0].boxes is not None:
                        scale_x = TARGET_WIDTH / (TARGET_WIDTH * PROCESSING_SCALE)
                        scale_y = TARGET_HEIGHT / (TARGET_HEIGHT * PROCESSING_SCALE)
                        
                        for result in vehicle_results[0].boxes:
                            x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
                            # Scale back to original frame size
                            x1, x2 = x1 * scale_x, x2 * scale_x
                            y1, y2 = y1 * scale_y, y2 * scale_y
                            conf = result.conf[0].cpu().item()
                            cls = int(result.cls[0].cpu().item())
                            detections.append([x1, y1, x2, y2, conf, cls])

                    # Process detections if any found
                    if detections:
                        dets = np.array(detections)
                        tracks = tracker.update(dets[:, :5])

                        for track in tracks:
                            track_id = int(track[4])
                            bbox = track[:4]
                            x1, y1, x2, y2 = map(int, bbox)

                            # Ensure bounding box is within frame bounds
                            x1 = max(0, min(x1, TARGET_WIDTH - 1))
                            y1 = max(0, min(y1, TARGET_HEIGHT - 1))
                            x2 = max(x1 + 1, min(x2, TARGET_WIDTH))
                            y2 = max(y1 + 1, min(y2, TARGET_HEIGHT))

                            # Find corresponding class
                            cls = None
                            for det in detections:
                                det_x1, det_y1, det_x2, det_y2, det_conf, det_cls = det
                                if not (x2 < det_x1 or x1 > det_x2 or y2 < det_y1 or y1 > det_y2):
                                    cls = det_cls
                                    break

                            if cls is not None:
                                track_class_labels[track_id] = cls

                            # Vehicle counting logic
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
                                        logger.info(f"Car entered: {car_count}")
                                    elif track_class_labels.get(track_id) == 3:
                                        bike_count += 1
                                        logger.info(f"Bike entered: {bike_count}")
                                    vehicle_directions[track_id] = 'crossed_down'
                                elif vehicle_directions[track_id] == 'down' and center_y < midline:
                                    if track_class_labels.get(track_id) == 2:
                                        car_count = max(0, car_count - 1)
                                        logger.info(f"Car exited: {car_count}")
                                    elif track_class_labels.get(track_id) == 3:
                                        bike_count = max(0, bike_count - 1)
                                        logger.info(f"Bike exited: {bike_count}")
                                    vehicle_directions[track_id] = 'crossed_up'

                            # License plate detection (even less frequent)
                            if track_id not in tracked_vehicles and frame_count % 30 == 0:
                                try:
                                    vehicle_crop = frame[y1:y2, x1:x2]
                                    if vehicle_crop.size > 0:
                                        license_plate_results = license_plate_model(vehicle_crop, conf=0.4)
                                        if (hasattr(license_plate_results[0], 'boxes') and 
                                            license_plate_results[0].boxes is not None):
                                            for lp_result in license_plate_results[0].boxes:
                                                lp_x1, lp_y1, lp_x2, lp_y2 = lp_result.xyxy[0].cpu().numpy()
                                                lp_x1 += x1
                                                lp_y1 += y1
                                                lp_x2 += x1
                                                lp_y2 += y1

                                                license_text = read_license_plate(frame, [lp_x1, lp_y1, lp_x2, lp_y2])
                                                tracked_vehicles[track_id] = license_text or 'N/A'
                                                break
                                except Exception as e:
                                    logger.error(f"License plate processing error: {str(e)}")
                                    tracked_vehicles[track_id] = 'N/A'

                            # Draw bounding box and labels on the display frame
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            class_names = {2: 'Car', 3: 'Bike', 5: 'Bus', 7: 'Truck'}
                            class_name = class_names.get(track_class_labels.get(track_id, 0), 'Vehicle')
                            
                            cv2.putText(frame, f"ID:{track_id} {class_name}", 
                                       (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            cv2.putText(frame, f"Plate: {tracked_vehicles.get(track_id, 'N/A')}", 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                except Exception as e:
                    logger.error(f"Error in YOLO processing: {str(e)}")

            # Always draw overlays
            cv2.line(frame, (0, midline), (TARGET_WIDTH, midline), (255, 0, 0), 3)
            cv2.putText(frame, f"Cars: {car_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Bikes: {bike_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add connection status with better visibility
            status_color = (0, 255, 0) if camera_connected else (0, 0, 255)
            cv2.putText(frame, f"RTSP: {'Connected' if camera_connected else 'Disconnected'}", 
                       (10, TARGET_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Encode with higher quality for smoother video
            ret_encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ret_encode and buffer is not None:
                with lock:
                    latest_frame = buffer.tobytes()
                        
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            consecutive_failures += 1
            time.sleep(0.02)  # Very short sleep to maintain frame rate

# Start frame processing thread
threading.Thread(target=process_frames, daemon=True).start()

def generate_frames():
    """Optimized frame generation for smoother streaming"""
    frame_delay = 0.033  # Target ~30 FPS
    
    while True:
        try:
            with lock:
                if latest_frame is None:
                    placeholder = create_placeholder_frame("Initializing camera...")
                    _, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    frame_data = buffer.tobytes() if buffer is not None else b''
                else:
                    frame_data = latest_frame
            
            if frame_data:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            
            time.sleep(frame_delay)
            
        except Exception as e:
            logger.error(f"Error in frame generation: {str(e)}")
            time.sleep(0.05)

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
    return jsonify({"cars": max(0, car_count), "bikes": max(0, bike_count)})

@app.route("/camera_status")
def camera_status():
    """Returns detailed camera connection status"""
    global cap, camera_connected, last_successful_frame_time
    
    is_connected = cap is not None and cap.isOpened() and camera_connected
    time_since_last_frame = time.time() - last_successful_frame_time
    
    return jsonify({
        "connected": is_connected,
        "resolution": f"{frame_width}x{frame_height}" if is_connected else "N/A",
        "last_frame_age": round(time_since_last_frame, 1),
        "status": "Active" if time_since_last_frame < 5 else "Stale"
    })

@app.route("/reset_counts")
def reset_counts():
    """Reset vehicle counts"""
    global car_count, bike_count
    car_count = 0
    bike_count = 0
    logger.info("Vehicle counts reset")
    return jsonify({"status": "Counts reset", "cars": car_count, "bikes": bike_count})

@app.route("/reconnect_camera")
def reconnect_camera_endpoint():
    """Force camera reconnection"""
    global connection_retry_count
    connection_retry_count = 0
    success = reconnect_camera()
    return jsonify({
        "status": "success" if success else "failed",
        "message": "Camera reconnection " + ("successful" if success else "failed")
    })

@app.route("/debug_rtsp")
def debug_rtsp():
    """Debug RTSP connection issues"""
    debug_info = {
        "opencv_version": cv2.__version__,
        "rtsp_url": rtsp_link,
        "target_resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        "processing_scale": PROCESSING_SCALE,
        "process_every_n_frames": PROCESS_EVERY_N_FRAMES,
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
            test_cap = cv2.VideoCapture(rtsp_link, backend)
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
    
    return jsonify(debug_info)

@app.route("/test_simple_rtsp")
def test_simple_rtsp():
    """Simple RTSP connection test"""
    try:
        test_cap = cv2.VideoCapture(rtsp_link)
        
        if not test_cap.isOpened():
            test_cap.release()
            return jsonify({
                "success": False,
                "error": "Could not open RTSP stream",
                "url": rtsp_link
            })
        
        ret, frame = test_cap.read()
        test_cap.release()
        
        if ret and frame is not None:
            height, width = frame.shape[:2]
            mean_pixel = float(np.mean(frame))
            
            return jsonify({
                "success": True,
                "frame_shape": [height, width],
                "mean_pixel_value": mean_pixel,
                "url": rtsp_link
            })
        else:
            return jsonify({
                "success": False,
                "error": "Could not read frame from RTSP stream",
                "url": rtsp_link
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "url": rtsp_link
        })

if __name__ == "__main__":
    try:
        logger.info("Starting Optimized Smart Parking System...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()