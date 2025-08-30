from flask import Flask, render_template, Response, jsonify
from flask import Flask, render_template, Response, jsonify
import cv2
import logging
import threading

from config.config import settings
from core.video_processor import SmartParkingProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

processor = SmartParkingProcessor()
processor.start_processing_thread() # Start the background processing thread

@app.route("/")
def home():
    """Serves the dashboard template"""
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """Serves the live video stream"""
    def generate_frames():
        frame_delay = 1.0 / 30.0 # Target ~30 FPS
        while True:
            frame_data = processor.get_current_frame_bytes()
            if frame_data:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            time.sleep(frame_delay)

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_counts")
def get_counts():
    """Returns live vehicle counts"""
    counts = processor.get_counts()
    return jsonify(counts)

@app.route("/camera_status")
def camera_status():
    """Returns detailed camera connection status"""
    status = processor.get_camera_status()
    return jsonify(status)

@app.route("/reset_counts")
def reset_counts():
    """Reset vehicle counts"""
    processor.reset_counts()
    counts = processor.get_counts()
    return jsonify({"status": "Counts reset", "cars": counts['cars'], "bikes": counts['bikes']})

@app.route("/reconnect_camera")
def reconnect_camera_endpoint():
    """Force camera reconnection"""
    success = processor.reconnect_camera()
    return jsonify({
        "status": "success" if success else "failed",
        "message": "Camera reconnection " + ("successful" if success else "failed")
    })

@app.route("/debug_rtsp")
def debug_rtsp():
    """Debug RTSP connection issues"""
    debug_info = processor.debug_rtsp_info()
    return jsonify(debug_info)

@app.route("/test_simple_rtsp")
def test_simple_rtsp():
    """Simple RTSP connection test"""
    test_result = processor.test_simple_rtsp_connection()
    return jsonify(test_result)

if __name__ == "__main__":
    try:
        logger.info("Starting Optimized Smart Parking System...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if processor.cap is not None:
            processor.cap.release()
        cv2.destroyAllWindows()
