from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from detect_utils import BallTracker
import uvicorn
import numpy as np
import cv2
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Golf Ball Tracker API")

# Initialize the ball tracker
try:
    # Get the absolute path to the model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best.pt')
    logger.info(f"Loading model from: {model_path}")
    tracker = BallTracker(model_path=model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_image(file: UploadFile) -> np.ndarray:
    """Read image file and convert to OpenCV format."""
    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image")
            raise HTTPException(status_code=400, detail="Could not read image")
        return img
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    """
    Analyze a frame to detect golf ball and determine its direction.
    Returns the processed image (base64) and direction.
    """
    try:
        logger.info("Received request to analyze frame")
        # Read and process the image
        frame = read_image(file)
        logger.info(f"Image loaded successfully. Shape: {frame.shape if frame is not None else 'None'}")
        
        processed_img, direction = tracker.process_frame(frame)
        logger.info(f"Frame processed. Direction: {direction}")
        
        return {
            "status": "success",
            "processed_image": processed_img,
            "direction": direction
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_frame: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)