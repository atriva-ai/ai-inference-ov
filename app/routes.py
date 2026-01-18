from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services import run_object_detection, draw_detections_on_image
from app.models import model_manager, ACCELERATORS, MODEL_NAME_MAPPING
from app.model_capabilities import get_detailed_model_info, get_available_model_types, get_available_objects as get_capability_objects, get_model_capabilities
from app.shared_data import (
    list_available_cameras, 
    get_frame_info, 
    get_latest_frame, 
    read_frame_file,
    list_camera_frames,
    get_shared_frames_path
)
from fastapi.responses import FileResponse
import os
from threading import Thread, Lock
from typing import Optional
import time
from pathlib import Path

router = APIRouter()

ARCHITECTURE = "openvino"

# Track continuous inference tasks
continuous_inference_tasks = {}
inference_lock = Lock()

@router.get("/models")
async def list_available_models():
    """Returns a list of all available models."""
    return {"available_models": model_manager.list_models()}

@router.get("/objects")
async def list_available_objects():
    """Returns a list of all available object types for detection."""
    return {"available_objects": get_capability_objects()}

@router.post("/inference/detection")
def detect_objects(object_name: str, image: UploadFile = File(...)):
    image_bytes = image.file.read()
    detections = run_object_detection(image_bytes, object_name)
    return {"objects": detections}

@router.get("/shared/cameras")
async def list_cameras():
    """List all cameras that have decoded frames available."""
    cameras = list_available_cameras()
    return {"cameras": cameras}

@router.get("/shared/cameras/{camera_id}/frames")
async def get_camera_frames(camera_id: str):
    """Get information about decoded frames for a specific camera."""
    frame_info = get_frame_info(camera_id)
    return frame_info

@router.get("/shared/cameras/{camera_id}/frames/latest")
async def get_camera_latest_frame(camera_id: str):
    """Get the latest decoded frame for a camera."""
    latest_frame = get_latest_frame(camera_id)
    if not latest_frame:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")
    
    return FileResponse(latest_frame)

@router.post("/shared/cameras/{camera_id}/inference")
async def detect_objects_in_camera_frame(camera_id: str, object_name: str):
    """Run object detection on the latest frame from a camera."""
    latest_frame = get_latest_frame(camera_id)
    if not latest_frame:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")
    
    # Read the frame file
    frame_bytes = read_frame_file(latest_frame)
    if not frame_bytes:
        raise HTTPException(status_code=500, detail=f"Failed to read frame file: {latest_frame}")
    
    # Run object detection
    detections = run_object_detection(frame_bytes, object_name)
    
    return {
        "camera_id": camera_id,
        "frame_path": latest_frame,
        "object_name": object_name,
        "detections": detections
    }

@router.get("/shared/cameras/{camera_id}/frames/{frame_index}")
async def get_camera_frame_by_index(camera_id: str, frame_index: int):
    """Get a specific frame by index for a camera."""
    frame_files = list_camera_frames(camera_id)
    if not frame_files:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")
    
    if frame_index < 0 or frame_index >= len(frame_files):
        raise HTTPException(
            status_code=400, 
            detail=f"Frame index {frame_index} out of range. Available frames: 0-{len(frame_files)-1}"
        )
    
    return FileResponse(frame_files[frame_index])

# --- Model Info API ---
@router.get("/model/info")
async def get_model_info():
    return {
        "models": get_detailed_model_info(),
        "model_types": get_available_model_types(),
        "objects": get_capability_objects(),  # Legacy compatibility
        "accelerators": ACCELERATORS,
        "architecture": ARCHITECTURE
    }

# --- Model Load API ---
@router.post("/model/load")
async def load_model(model_name: str, accelerator: Optional[str] = "cpu32"):
    if accelerator not in ACCELERATORS:
        raise HTTPException(status_code=400, detail=f"Unsupported accelerator: {accelerator}")
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    try:
        # Create a new manager for the requested accelerator
        from app.models import ModelManager
        manager = ModelManager(acceleration=accelerator)
        compiled_model, input_shape = manager.load_model(model_name)
        return {
            "model_name": model_name,
            "accelerator": accelerator,
            "architecture": ARCHITECTURE,
            "status": "loaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Direct Inference API ---
@router.post("/inference/direct")
async def direct_inference(model_name: str, image: UploadFile = File(...)):
    """Run direct inference using a specific model."""
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    
    try:
        from app.services import preprocess_image, run_inference
        
        image_bytes = image.file.read()
        
        # Load model and get input shape
        compiled_model, input_shape = model_manager.load_model(model_name)
        
        # Preprocess image (returns tuple: input_data, original_shape, ratio, pad)
        preprocessed_result = preprocess_image(image_bytes, input_shape)
        input_data = preprocessed_result[0]  # Extract just the input data
        
        # Run inference
        output = run_inference(input_data, compiled_model)
        
        return {
            "model_name": model_name,
            "input_shape": list(input_shape) if hasattr(input_shape, '__iter__') else input_shape,
            "output_shape": list(output.shape) if hasattr(output, 'shape') else None,
            "output": output.tolist() if hasattr(output, 'tolist') else str(output)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Inference on Latest Frame API ---
@router.post("/inference/latest-frame")
async def inference_latest_frame(camera_id: str, model_name: str, accelerator: Optional[str] = "cpu32"):
    if accelerator not in ACCELERATORS:
        raise HTTPException(status_code=400, detail=f"Unsupported accelerator: {accelerator}")
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    latest_frame = get_latest_frame(camera_id)
    if not latest_frame:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")
    frame_bytes = read_frame_file(latest_frame)
    if not frame_bytes:
        raise HTTPException(status_code=500, detail=f"Failed to read frame file: {latest_frame}")
    from app.models import ModelManager
    manager = ModelManager(acceleration=accelerator)
    from app.services import preprocess_image, run_inference
    compiled_model, input_shape = manager.load_model(model_name)
    image = preprocess_image(frame_bytes, input_shape)
    model_output = run_inference(image, compiled_model)
    # Parse output (same as run_object_detection)
    confidence_threshold = 0.3
    detections = []
    for detection in model_output:
        image_id, class_id, confidence, xmin, ymin, xmax, ymax = detection
        if confidence > confidence_threshold:
            detections.append({
                "class_id": int(class_id),
                "confidence": float(confidence),
                "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
            })
    return {
        "camera_id": camera_id,
        "model_name": model_name,
        "accelerator": accelerator,
        "architecture": ARCHITECTURE,
        "frame_path": latest_frame,
        "detections": detections
    }

# --- Background Inference API ---
def background_inference(camera_id: str, model_name: str, accelerator: str):
    from app.models import ModelManager
    from app.services import preprocess_image, run_inference
    manager = ModelManager(acceleration=accelerator)
    compiled_model, input_shape = manager.load_model(model_name)
    frame_files = list_camera_frames(camera_id)
    results = []
    for frame_path in frame_files:
        frame_bytes = read_frame_file(frame_path)
        if not frame_bytes:
            continue
        image = preprocess_image(frame_bytes, input_shape)
        model_output = run_inference(image, compiled_model)
        confidence_threshold = 0.3
        detections = []
        for detection in model_output:
            image_id, class_id, confidence, xmin, ymin, xmax, ymax = detection
            if confidence > confidence_threshold:
                detections.append({
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
                })
        results.append({
            "frame_path": frame_path,
            "detections": detections
        })
    # Optionally: save results to a file or database
    print(f"Background inference for camera {camera_id} complete. {len(results)} frames processed.")

@router.post("/inference/background")
async def start_background_inference(camera_id: str, model_name: str, accelerator: Optional[str] = "cpu32"):
    if accelerator not in ACCELERATORS:
        raise HTTPException(status_code=400, detail=f"Unsupported accelerator: {accelerator}")
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    thread = Thread(target=background_inference, args=(camera_id, model_name, accelerator), daemon=True)
    thread.start()
    return {
        "camera_id": camera_id,
        "model_name": model_name,
        "accelerator": accelerator,
        "architecture": ARCHITECTURE,
        "status": "background_inference_started"
    }

# --- Continuous Inference API ---
def continuous_inference_worker(camera_id: str, model_name: str, accelerator: str, object_filter: Optional[str] = None):
    """Continuous inference worker that processes new frames as they arrive."""
    from app.models import ModelManager
    from app.services import preprocess_image, run_inference, postprocess_detections
    
    print(f"🤖 Starting continuous inference for camera {camera_id} with model {model_name}")
    
    try:
        manager = ModelManager(acceleration=accelerator)
        compiled_model, input_shape = manager.load_model(model_name)
        
        # Get model configuration
        model_config = manager.get_model_config(model_name)
        classes = model_config.get("classes", [])
        confidence_threshold = model_config.get("confidence_threshold", 0.25)
        nms_threshold = model_config.get("nms_threshold", 0.45)
        model_type = model_config.get("model_type", "object_detection")
        
        # Filter for person class if object_filter is "person"
        # Person is class 0 in COCO dataset (YOLOv8)
        target_class_id = None
        if object_filter == "person":
            # Person is always class 0 in COCO dataset
            target_class_id = 0
        
        frames_dir = get_shared_frames_path() / camera_id
        annotated_dir = frames_dir / "annotated"
        annotated_dir.mkdir(exist_ok=True)
        
        last_processed_frame = None
        
        while True:
            with inference_lock:
                if camera_id not in continuous_inference_tasks:
                    print(f"🛑 Stopping continuous inference for camera {camera_id}")
                    break
                task_info = continuous_inference_tasks.get(camera_id)
                if not task_info or not task_info.get("running", False):
                    print(f"🛑 Continuous inference stopped for camera {camera_id}")
                    break
            
            # Get latest frame
            latest_frame = get_latest_frame(camera_id)
            if not latest_frame:
                time.sleep(1)
                continue
            
            # Skip if we already processed this frame
            if latest_frame == last_processed_frame:
                time.sleep(0.5)
                continue
            
            try:
                # Read frame
                frame_bytes = read_frame_file(latest_frame)
                if not frame_bytes:
                    time.sleep(0.5)
                    continue
                
                # Run inference
                use_letterbox = "yolo" in model_name.lower() or model_type == "object_detection"
                image_data, original_shape, ratio, pad = preprocess_image(frame_bytes, input_shape, use_letterbox)
                model_output = run_inference(image_data, compiled_model)
                
                # Postprocess detections
                detections = postprocess_detections(
                    model_output, original_shape, ratio, pad, classes,
                    confidence_threshold, nms_threshold, model_type
                )
                
                # Filter for person class if specified
                if target_class_id is not None:
                    detections = [d for d in detections if d.get('class_id') == target_class_id]
                
                # Draw detections on image
                annotated_bytes = draw_detections_on_image(frame_bytes, detections)
                
                # Save annotated frame
                frame_path = Path(latest_frame)
                annotated_path = annotated_dir / f"annotated_{frame_path.name}"
                with open(annotated_path, 'wb') as f:
                    f.write(annotated_bytes)
                
                last_processed_frame = latest_frame
                print(f"✅ Processed frame for camera {camera_id}: {len(detections)} detections")
                
            except Exception as e:
                print(f"❌ Error processing frame for camera {camera_id}: {e}")
                time.sleep(1)
            
            time.sleep(0.5)  # Check for new frames every 0.5 seconds
            
    except Exception as e:
        print(f"❌ Continuous inference error for camera {camera_id}: {e}")
    finally:
        with inference_lock:
            if camera_id in continuous_inference_tasks:
                continuous_inference_tasks[camera_id]["running"] = False

@router.post("/inference/continuous/start")
async def start_continuous_inference(
    camera_id: str,
    model_name: str = "yolov8n",
    accelerator: Optional[str] = "cpu32",
    object_filter: Optional[str] = "person"
):
    """Start continuous inference for a camera."""
    if accelerator not in ACCELERATORS:
        raise HTTPException(status_code=400, detail=f"Unsupported accelerator: {accelerator}")
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    
    with inference_lock:
        if camera_id in continuous_inference_tasks and continuous_inference_tasks[camera_id].get("running", False):
            return {
                "camera_id": camera_id,
                "status": "already_running",
                "message": "Continuous inference already running for this camera"
            }
        
        continuous_inference_tasks[camera_id] = {
            "running": True,
            "model_name": model_name,
            "accelerator": accelerator,
            "object_filter": object_filter
        }
    
    thread = Thread(target=continuous_inference_worker, args=(camera_id, model_name, accelerator, object_filter), daemon=True)
    thread.start()
    
    return {
        "camera_id": camera_id,
        "model_name": model_name,
        "accelerator": accelerator,
        "object_filter": object_filter,
        "status": "started"
    }

@router.post("/inference/continuous/stop")
async def stop_continuous_inference(camera_id: str):
    """Stop continuous inference for a camera."""
    with inference_lock:
        if camera_id in continuous_inference_tasks:
            continuous_inference_tasks[camera_id]["running"] = False
            del continuous_inference_tasks[camera_id]
            return {"camera_id": camera_id, "status": "stopped"}
        else:
            return {"camera_id": camera_id, "status": "not_running"}

@router.get("/inference/continuous/status")
async def get_continuous_inference_status(camera_id: str):
    """Get status of continuous inference for a camera."""
    with inference_lock:
        task = continuous_inference_tasks.get(camera_id)
        if task:
            return {
                "camera_id": camera_id,
                "running": task.get("running", False),
                "model_name": task.get("model_name"),
                "accelerator": task.get("accelerator"),
                "object_filter": task.get("object_filter")
            }
        else:
            return {
                "camera_id": camera_id,
                "running": False
            }

@router.get("/shared/cameras/{camera_id}/frames/latest/annotated")
async def get_camera_latest_annotated_frame(camera_id: str):
    """Get the latest annotated frame for a camera."""
    frames_dir = get_shared_frames_path() / camera_id
    annotated_dir = frames_dir / "annotated"
    
    if not annotated_dir.exists():
        raise HTTPException(status_code=404, detail=f"No annotated frames found for camera {camera_id}")
    
    # Get all annotated frames
    annotated_frames = sorted(annotated_dir.glob("annotated_*.jpg"))
    if not annotated_frames:
        raise HTTPException(status_code=404, detail=f"No annotated frames available for camera {camera_id}")
    
    # Return the latest annotated frame
    latest_annotated = annotated_frames[-1]
    return FileResponse(latest_annotated)

@router.get("/shared/cameras/{camera_id}/detections/latest")
async def get_camera_latest_detections(camera_id: str):
    """Get the latest detection data with bounding boxes for a camera."""
    from app.models import ModelManager
    from app.services import preprocess_image, run_inference, postprocess_detections
    
    # Get latest frame
    latest_frame = get_latest_frame(camera_id)
    if not latest_frame:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")
    
    # Check if continuous inference is running for this camera
    with inference_lock:
        task = continuous_inference_tasks.get(camera_id)
        if not task or not task.get("running", False):
            raise HTTPException(status_code=404, detail=f"Continuous inference not running for camera {camera_id}")
        
        model_name = task.get("model_name", "yolov8n")
        accelerator = task.get("accelerator", "cpu32")
        object_filter = task.get("object_filter")
    
    try:
        # Read frame
        frame_bytes = read_frame_file(latest_frame)
        if not frame_bytes:
            raise HTTPException(status_code=500, detail=f"Failed to read frame file: {latest_frame}")
        
        # Run inference to get detections
        manager = ModelManager(acceleration=accelerator)
        compiled_model, input_shape = manager.load_model(model_name)
        model_config = manager.get_model_config(model_name)
        classes = model_config.get("classes", [])
        confidence_threshold = model_config.get("confidence_threshold", 0.25)
        nms_threshold = model_config.get("nms_threshold", 0.45)
        model_type = model_config.get("model_type", "object_detection")
        
        use_letterbox = "yolo" in model_name.lower() or model_type == "object_detection"
        image_data, original_shape, ratio, pad = preprocess_image(frame_bytes, input_shape, use_letterbox)
        model_output = run_inference(image_data, compiled_model)
        
        # Postprocess detections
        detections = postprocess_detections(
            model_output, original_shape, ratio, pad, classes,
            confidence_threshold, nms_threshold, model_type
        )
        
        # Filter for person class if specified
        if object_filter == "person":
            target_class_id = 0  # Person is class 0 in COCO dataset
            detections = [d for d in detections if d.get('class_id') == target_class_id]
        
        # Format detections with bounding boxes
        formatted_detections = []
        for det in detections:
            formatted_detections.append({
                "class_id": det.get("class_id"),
                "class_name": det.get("class_name"),
                "confidence": det.get("confidence"),
                "bbox": det.get("bbox_xyxy", [])  # [x1, y1, x2, y2] format
            })
        
        return {
            "camera_id": camera_id,
            "frame_path": latest_frame,
            "detections": formatted_detections,
            "detection_count": len(formatted_detections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get detections: {str(e)}")

# --- Model Capabilities API ---
@router.get("/models/{model_name}/capabilities")
async def get_model_capabilities_endpoint(model_name: str):
    """Get detailed capabilities for a specific model."""
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    
    capabilities = get_model_capabilities(model_name)
    return {
        "model_name": model_name,
        "capabilities": capabilities,
        "architecture": ARCHITECTURE
    }

