from flask import Flask, render_template, Response, jsonify, request
import logging
import threading
import os
import time
import signal
import sys
from contextlib import contextmanager

from config.config import settings
from core.optimized_processor import OptimizedParkingProcessor

# Ensure logs directory exists
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'optimized_app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Global processor instance
processor = None
shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_event.set()
    if processor:
        processor.cleanup()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@contextmanager
def get_processor():
    global processor
    if processor is None:
        try:
            processor = OptimizedParkingProcessor()
            processor.start_processing_thread()
            logger.info("Processor initialized and started")
        except Exception as e:
            logger.error(f"Failed to initialize processor: {e}")
            processor = None
            raise
    
    try:
        yield processor
    except Exception as e:
        logger.error(f"Processor error: {e}")
        if "critical" in str(e).lower():
            processor = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    def generate_frames():
        frame_delay = 1.0 / settings.get('target_fps', 30)
        consecutive_empty_frames = 0
        max_empty_frames = 10
        
        while not shutdown_event.is_set():
            try:
                with get_processor() as proc:
                    frame_data = proc.get_current_frame_bytes()
                    
                    if frame_data:
                        consecutive_empty_frames = 0
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                    else:
                        consecutive_empty_frames += 1
                        if consecutive_empty_frames > max_empty_frames:
                            logger.warning("Too many consecutive empty frames")
                            break
                
                time.sleep(frame_delay)
                
            except Exception as e:
                logger.error(f"Error in video feed generation: {e}")
                time.sleep(0.1)
    
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_counts")
def get_counts():
    try:
        with get_processor() as proc:
            counts = proc.get_counts()
            return jsonify(counts)
    except Exception as e:
        logger.error(f"Error getting counts: {e}")
        return jsonify({"cars": 0, "bikes": 0, "total": 0, "error": str(e)})

@app.route("/get_performance")
def get_performance():
    try:
        with get_processor() as proc:
            stats = proc.get_performance_stats()
            detector_stats = proc.detector.get_performance_stats()
            stats.update({
                'detector_fps': detector_stats.get('fps', 0),
                'detector_inference_time_ms': detector_stats.get('average_inference_time_ms', 0),
                'detector_device': detector_stats.get('device', 'unknown')
            })
            return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        return jsonify({"error": str(e)})

@app.route("/reset_counts", methods=['POST'])
def reset_counts():
    try:
        with get_processor() as proc:
            proc.reset_counts()
            counts = proc.get_counts()
            return jsonify({
                "status": "success",
                "message": "Counts reset successfully",
                "counts": counts
            })
    except Exception as e:
        logger.error(f"Error resetting counts: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health")
def health_check():
    try:
        with get_processor() as proc:
            camera_status = proc.camera_connected
            performance = proc.get_performance_stats()
            
            return jsonify({
                "status": "healthy" if camera_status else "degraded",
                "camera_connected": camera_status,
                "fps": performance.get('fps', 0),
                "memory_usage_mb": performance.get('memory_usage_mb', 0),
                "uptime": time.time() - app.start_time if hasattr(app, 'start_time') else 0
            })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/config")
def get_config():
    safe_config = {
        'target_width': settings.get('target_width'),
        'target_height': settings.get('target_height'),
        'process_every_n_frames': settings.get('process_every_n_frames'),
        'confidence_threshold': settings.get('confidence_threshold'),
        'max_detections': settings.get('max_detections'),
        'target_fps': settings.get('target_fps'),
        'has_rtsp': bool(settings.get('rtsp_link')),
        'has_local_video': bool(settings.get('video_path'))
    }
    return jsonify(safe_config)

@app.route("/reconnect", methods=['POST'])
def reconnect_camera():
    try:
        with get_processor() as proc:
            success = proc.initialize_camera()
            return jsonify({
                "status": "success" if success else "failed",
                "message": "Camera reconnection " + ("successful" if success else "failed")
            })
    except Exception as e:
        logger.error(f"Error reconnecting camera: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

def initialize_app():
    global processor
    
    try:
        processor = OptimizedParkingProcessor()
        processor.start_processing_thread()
        
        app.start_time = time.time()
        
        logger.info("Optimized Flask application initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        return False

if __name__ == "__main__":
    try:
        logger.info("Starting Optimized Smart Parking System...")
        
        if not initialize_app():
            logger.error("Failed to initialize application")
            sys.exit(1)
        
        # Run the application
        app.run(
            host='0.0.0.0', 
            port=5000, 
            debug=False, 
            threaded=True,
            use_reloader=False  # Disable reloader to avoid duplicate processes
        )
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        logger.info("Shutting down...")
        shutdown_event.set()
        if processor:
            processor.cleanup()
        logger.info("Shutdown complete")
