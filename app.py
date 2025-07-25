from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from detect_utils import BallTracker
import numpy as np
import cv2
import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ball Tracker API")

# Initialize the ball tracker
try:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best.pt')
    logger.info(f"Loading model from: {model_path}")
    tracker = BallTracker(model_path=model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

async def process_uploaded_file(file: UploadFile) -> np.ndarray:
    """Process uploaded file into OpenCV image format."""
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file")
        
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image format")
    return img

@app.post("/analyze")
async def analyze_frame(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Process frame and return ball detection results."""
    try:
        # Process the uploaded file
        frame = await process_uploaded_file(file)
        
        # Run detection
        processed_img, direction = tracker.process_frame(frame)
        
        return {
            "status": "success",
            "direction": direction,
            "processed_image": processed_img
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)