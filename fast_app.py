from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import threading
from config.config import settings
from core.video_processor import SmartParkingProcessor

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

processor = SmartParkingProcessor()
cap = cv2.VideoCapture(settings.get('video_path'))

lock = threading.Lock()
latest_frame = None

def process_frames_fastapi():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        processed_frame = processor.process_frame(frame)

        _, buffer = cv2.imencode('.jpg', processed_frame)
        with lock:
            latest_frame = buffer.tobytes()

threading.Thread(target=process_frames_fastapi, daemon=True).start()

def generate_frames_fastapi():
    while True:
        with lock:
            if latest_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames_fastapi(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/get_counts")
def get_counts():
    return processor.get_counts()
