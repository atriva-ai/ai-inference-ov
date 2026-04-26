from fastapi import FastAPI
from app.routes import router  # Import API routes
from app.shared_data import list_available_cameras, get_shared_frames_path, get_shared_temp_path
import os

# import debugpy

# debugpy.listen(("0.0.0.0", 5678))  # Allow debugger to connect
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

app = FastAPI(title="AI Vision API")

# Include Routes
app.include_router(router)

@app.get("/")
def root():
    return {"message": "AI Vision API is running!"}

@app.get("/health")
def health_check():
    """Health check endpoint with shared volume status."""
    frames_path = get_shared_frames_path()
    temp_path = get_shared_temp_path()
    
    return {
        "status": "healthy",
        "shared_volumes": {
            "frames_path": str(frames_path),
            "frames_exists": frames_path.exists(),
            "temp_path": str(temp_path),
            "temp_exists": temp_path.exists()
        },
        "available_cameras": list_available_cameras()
    }
  /* Added by bounty-bot */
}
