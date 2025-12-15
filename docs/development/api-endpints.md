# 🧩 Atriva AI OpenVINO Inference API — Endpoints

This document provides an overview of the REST API endpoints exposed by the Atriva AI service built on OpenVINO. Use these endpoints to query models, run inference, and interact with the AI runtime from any HTTP client or application.

- **Base URL**: By default, the API runs at `http://localhost:8000`
- **Interactive Docs**: Visit [`/docs`](http://localhost:8000/docs) for an OpenAPI/Swagger UI
- **Version**: Compatible with Atriva AI Runtime v1.0+ and OpenVINO 2023.3+

These endpoints allow you to:
- Query health, models, and available objects
- Load models and change accelerator devices
- Submit images for detection/inference
- Access shared video frames and perform camera-based inference

---


## **🛠 API**

### **Overview**

The API section documents the RESTful endpoints available in the Atriva AI API. The service provides a FastAPI-based interface for running AI inference tasks, managing models, and accessing camera frames. All endpoints are documented with interactive Swagger UI available at `/docs` when the service is running.

The API is organized into several categories:
- **System & Health**: Basic health checks and system status
- **Model Information**: Query available models, objects, and capabilities
- **Model Management**: Load models with specific accelerators
- **Inference**: Run object detection and AI inference on images or camera frames
- **Camera & Shared Frames**: Access decoded camera frames and run inference on them

### **System & Health Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root endpoint - returns API status message |
| `GET` | `/health` | Health check with shared volume status and available cameras |

**Example Response (`/health`):**
```json
{
  "status": "healthy",
  "shared_volumes": {
    "frames_path": "/path/to/frames",
    "frames_exists": true,
    "temp_path": "/path/to/temp",
    "temp_exists": true
  },
  "available_cameras": ["camera1", "camera2"]
}
```

### **Model Information Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/models` | List all available models |
| `GET` | `/objects` | List all available object types for detection |
| `GET` | `/model/info` | Get detailed model information, types, objects, accelerators, and architecture |
| `GET` | `/models/{model_name}/capabilities` | Get detailed capabilities for a specific model |

**Example Response (`/models`):**
```json
{
  "available_models": ["yolov8n", "yolov8s", "yolov8m", "vehicle", "car", "person", ...]
}
```

**Example Response (`/model/info`):**
```json
{
  "models": {...},
  "model_types": ["detection", "classification"],
  "objects": ["car", "person", "bicycle", ...],
  "accelerators": ["cpui8", "cpu16", "cpu32"],
  "architecture": "openvino"
}
```

**Available Accelerators:**
- `cpui8` - CPU with INT8 precision
- `cpu16` - CPU with FP16 precision
- `cpu32` - CPU with FP32 precision (default)

### **Model Management Endpoints**

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| `POST` | `/model/load` | Load a model with a specific accelerator | `model_name` (required), `accelerator` (optional, default: "cpu32") |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/model/load?model_name=yolov8n&accelerator=cpu32"
```

**Example Response:**
```json
{
  "model_name": "yolov8n",
  "accelerator": "cpu32",
  "architecture": "openvino",
  "status": "loaded"
}
```

### **Inference Endpoints**

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| `POST` | `/inference/detection` | Detect objects in an uploaded image | `object_name` (query param), `image` (file upload) |
| `POST` | `/inference/direct` | Run direct inference using a specific model | `model_name` (query param), `image` (file upload) |
| `POST` | `/inference/latest-frame` | Run inference on the latest frame from a camera | `camera_id` (query param), `model_name` (query param), `accelerator` (optional) |
| `POST` | `/inference/background` | Start background inference on all frames from a camera | `camera_id` (query param), `model_name` (query param), `accelerator` (optional) |

**Example Request (`/inference/detection`):**
```bash
curl -X POST "http://localhost:8000/inference/detection?object_name=car" \
  -F "image=@test_image.jpg"
```

**Example Response:**
```json
{
  "objects": [
    {
      "class_id": 2,
      "confidence": 0.95,
      "bbox": [100, 150, 300, 400]
    }
  ]
}
```

**Example Request (`/inference/direct`):**
```bash
curl -X POST "http://localhost:8000/inference/direct?model_name=yolov8n" \
  -F "image=@test_image.jpg"
```

**Example Response:**
```json
{
  "model_name": "yolov8n",
  "input_shape": [1, 3, 640, 640],
  "output_shape": [1, 84, 8400],
  "output": [[...]]
}
```

### **Camera & Shared Frame Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/shared/cameras` | List all cameras that have decoded frames available |
| `GET` | `/shared/cameras/{camera_id}/frames` | Get information about decoded frames for a specific camera |
| `GET` | `/shared/cameras/{camera_id}/frames/latest` | Get the latest decoded frame image for a camera |
| `GET` | `/shared/cameras/{camera_id}/frames/{frame_index}` | Get a specific frame by index for a camera |
| `POST` | `/shared/cameras/{camera_id}/inference` | Run object detection on the latest frame from a camera | `object_name` (query param) |

**Example Request (`/shared/cameras`):**
```bash
curl "http://localhost:8000/shared/cameras"
```

**Example Response:**
```json
{
  "cameras": ["camera1", "camera2"]
}
```

**Example Request (`/shared/cameras/{camera_id}/inference`):**
```bash
curl -X POST "http://localhost:8000/shared/cameras/camera1/inference?object_name=car"
```

**Example Response:**
```json
{
  "camera_id": "camera1",
  "frame_path": "/path/to/frame.jpg",
  "object_name": "car",
  "detections": [
    {
      "class_id": 2,
      "confidence": 0.92,
      "bbox": [150, 200, 350, 450]
    }
  ]
}
```

### **API Documentation**

When the service is running, you can access:
- **Interactive API Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Alternative API Documentation (ReDoc)**: `http://localhost:8000/redoc`
