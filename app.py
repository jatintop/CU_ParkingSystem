import cv2
from flask import Flask, Response, render_template
from config.config import settings
from core.video_processor import SmartParkingProcessor

app = Flask(__name__)
processor = SmartParkingProcessor()

def process_video_stream():
    cap = cv2.VideoCapture(settings.get('video_path'))
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            continue

        processed_frame = processor.process_frame(frame)

        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(process_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_counts')
def get_counts():
    return processor.get_counts()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
