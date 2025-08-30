from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import threading
import os
import time
import signal
import sys
from contextlib import contextmanager
from typing import Dict, Any

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
        logging.FileHandler(os.path.join(log_dir, 'optimized_fastapi.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Optimized Smart Parking System",
    description="High-performance vehicle counting system",
    version="2.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
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
    
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/get_counts")
async def get_counts() -> Dict[str, Any]:
    try:
        with get_processor() as proc:
            counts = proc.get_counts()
            return counts
    except Exception as e:
        logger.error(f"Error getting counts: {e}")
        return {"cars": 0, "bikes": 0, "total": 0, "error": str(e)}

@app.get("/get_performance")
async def get_performance() -> Dict[str, Any]:
    try:
        with get_processor() as proc:
            stats = proc.get_performance_stats()
            detector_stats = proc.detector.get_performance_stats()
            stats.update({
                'detector_fps': detector_stats.get('fps', 0),
                'detector_inference_time_ms': detector_stats.get('average_inference_time_ms', 0),
                'detector_device': detector_stats.get('device', 'unknown')
            })
            return stats
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_counts")
async def reset_counts() -> Dict[str, Any]:
    try:
        with get_processor() as proc:
            proc.reset_counts()
            counts = proc.get_counts()
            return {
                "status": "success",
                "message": "Counts reset successfully",
                "counts": counts
            }
    except Exception as e:
        logger.error(f"Error resetting counts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    try:
        with get_processor() as proc:
            camera_status = proc.camera_connected
            performance = proc.get_performance_stats()
            
            return {
                "status": "healthy" if camera_status else "degraded",
                "camera_connected": camera_status,
                "fps": performance.get('fps', 0),
                "memory_usage_mb": performance.get('memory_usage_mb', 0),
                "uptime": time.time() - app.start_time if hasattr(app, 'start_time') else 0
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config() -> Dict[str, Any]:
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
    return safe_config

@app.post("/reconnect")
async def reconnect_camera() -> Dict[str, Any]:
    try:
        with get_processor() as proc:
            success = proc.initialize_camera()
            return {
                "status": "success" if success else "failed",
                "message": "Camera reconnection " + ("successful" if success else "failed")
            }
    except Exception as e:
        logger.error(f"Error reconnecting camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def api_status() -> Dict[str, Any]:
    try:
        with get_processor() as proc:
            counts = proc.get_counts()
            performance = proc.get_performance_stats()
            detector_stats = proc.detector.get_performance_stats()
            
            return {
                "system": {
                    "status": "running",
                    "uptime": time.time() - app.start_time if hasattr(app, 'start_time') else 0,
                    "version": "2.0.0"
                },
                "camera": {
                    "connected": proc.camera_connected,
                    "resolution": f"{proc.target_width}x{proc.target_height}"
                },
                "counts": counts,
                "performance": {
                    "fps": performance.get('fps', 0),
                    "memory_usage_mb": performance.get('memory_usage_mb', 0),
                    "cpu_usage_percent": performance.get('cpu_usage_percent', 0),
                    "detector_fps": detector_stats.get('fps', 0),
                    "detector_inference_time_ms": detector_stats.get('average_inference_time_ms', 0)
                }
            }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": request.url.path}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

def initialize_app():
    global processor
    
    try:
        processor = OptimizedParkingProcessor()
        processor.start_processing_thread()
        
        app.start_time = time.time()
        
        logger.info("Optimized FastAPI application initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Optimized Smart Parking System (FastAPI)...")
    if not initialize_app():
        logger.error("Failed to initialize application")
        sys.exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Optimized Smart Parking System...")
    shutdown_event.set()
    if processor:
        processor.cleanup()
    logger.info("Shutdown complete")

if __name__ == "__main__":
    import uvicorn
    
    try:
        uvicorn.run(
            "optimized_fastapi_app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload to avoid duplicate processes
            workers=1,     # Single worker for better resource management
            log_level="info"
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
