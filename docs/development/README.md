# Atriva AI Inference Service — Development Guide

This guide covers everything you need to develop, test, and extend the Atriva AI Inference Service built on OpenVINO.

---

## 1. Project Structure

```
atriva-ai-openvino/
├── app/
│   ├── __init__.py
│   ├── routes.py            # FastAPI endpoint definitions
│   ├── services.py          # Inference logic & model orchestration
│   ├── models.py            # Pydantic request/response schemas
│   ├── model_capabilities.py # Model metadata & supported objects
│   └── shared_data.py       # Shared volume frame access utilities
├── models/                  # OpenVINO IR model files (.xml, .bin)
│   ├── yolov8n/
│   ├── yolov8s/
│   ├── lprnet/
│   └── vehicle_tracking/
├── scripts/
│   ├── convert_to_openvino.py   # PyTorch → OpenVINO conversion
│   └── requirements.txt
├── tests/
│   ├── test_runner.py           # Main test orchestrator
│   ├── test_yolov8_openvino.py  # YOLOv8 detection tests
│   ├── test_vehicle_tracking.py # Tracking tests
│   ├── test_images/             # Sample images
│   ├── test_videos/             # Sample videos
│   └── requirements.txt
├── main.py                  # FastAPI application entry point
├── config.py                # Configuration settings
├── requirements.txt         # Production dependencies
└── Dockerfile               # Container build configuration
```

---

## 2. Environment Setup

### 2.1 Python Version Requirements

| Environment | Python Version | Reason |
|-------------|----------------|--------|
| Docker AI Service | 3.12 | Latest stable, handled by Dockerfile |
| Testing | **3.11** | PyTorch wheel availability |
| Model Conversion | **3.11** | PyTorch + OpenVINO compatibility |

> ⚠️ **Important**: PyTorch does not provide pre-built wheels for Python 3.13. Use Python 3.11 for any tasks involving PyTorch.

### 2.2 Local Development Setup

```bash
# Clone the repository
git clone https://github.com/atriva-ai/atriva-ai-openvino.git
cd atriva-ai-openvino

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify OpenVINO installation
python -c "from openvino import Core; print('OpenVINO OK:', Core().available_devices)"
```

### 2.3 IDE Configuration

**VS Code** (recommended extensions):
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Black Formatter (ms-python.black-formatter)

**PyCharm**:
- Set Python interpreter to `venv/bin/python`
- Enable FastAPI support in project settings

---

## 3. Running the Development Server

### 3.1 Start FastAPI with Hot Reload

```bash
# From project root
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# With debug logging
LOG_LEVEL=debug uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Access points:
- **API**: http://localhost:8001
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

### 3.2 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8001` | API server port |
| `LOG_LEVEL` | `info` | Logging verbosity (debug, info, warning, error) |
| `MODELS_PATH` | `./models` | Path to OpenVINO model files |
| `SHARED_FRAMES_PATH` | `/shared/frames` | Shared volume for video frames |
| `DEFAULT_ACCELERATOR` | `cpu32` | Default inference precision |

### 3.3 Configuration File

Edit `config.py` for persistent configuration:

```python
# config.py
import os

class Settings:
    MODELS_PATH = os.getenv("MODELS_PATH", "./models")
    SHARED_FRAMES_PATH = os.getenv("SHARED_FRAMES_PATH", "/shared/frames")
    DEFAULT_ACCELERATOR = os.getenv("DEFAULT_ACCELERATOR", "cpu32")
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
```

---

## 4. Development Workflow

### 4.1 Adding a New API Endpoint

1. **Define the route** in `app/routes.py`:

```python
from fastapi import APIRouter, Query
from app.models import InferenceResponse
from app.services import run_custom_inference

router = APIRouter()

@router.post("/inference/custom", response_model=InferenceResponse)
async def custom_inference(
    model_name: str = Query(..., description="Model to use"),
    threshold: float = Query(0.5, description="Confidence threshold")
):
    """Run custom inference with specified parameters."""
    result = await run_custom_inference(model_name, threshold)
    return result
```

2. **Add business logic** in `app/services.py`:

```python
async def run_custom_inference(model_name: str, threshold: float):
    # Load model
    model = get_or_load_model(model_name)
    
    # Run inference
    outputs = model.infer(inputs)
    
    # Post-process with threshold
    detections = post_process(outputs, threshold)
    
    return {"detections": detections}
```

3. **Define schemas** in `app/models.py`:

```python
from pydantic import BaseModel
from typing import List

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]

class InferenceResponse(BaseModel):
    model_name: str
    detections: List[Detection]
```

### 4.2 Adding a New Model

1. **Convert to OpenVINO IR format**:

```bash
cd scripts
source scripts-venv-py311/bin/activate

# For YOLO models
python convert_to_openvino.py --model yolov8n --output ../models/yolov8n

# For custom ONNX models
mo --input_model custom_model.onnx --output_dir ../models/custom
```

2. **Register model capabilities** in `app/model_capabilities.py`:

```python
MODEL_REGISTRY = {
    "custom_model": {
        "type": "detection",
        "input_size": (640, 640),
        "classes": ["class1", "class2", "class3"],
        "path": "models/custom/model.xml"
    }
}
```

3. **Test the model**:

```bash
curl -X POST "http://localhost:8001/inference/direct?model_name=custom_model" \
  -F "image=@test_image.jpg"
```

---

## 5. Testing

### 5.1 Test Environment Setup

```bash
cd tests

# Use Python 3.11 for PyTorch compatibility
pyenv local 3.11.13  # or: python3.11 -m venv test-venv-py311
python3.11 -m venv test-venv-py311
source test-venv-py311/bin/activate

pip install -r requirements.txt
```

### 5.2 Running Tests

```bash
# Run all tests
python test_runner.py

# Run specific model test
python test_runner.py --model yolov8n --input test_images/sample.jpg

# Run with video input
python test_runner.py --model vehicle_tracking --input test_videos/traffic.mp4

# Download models first (if needed)
python test_runner.py --download-models
```

### 5.3 Test Output

Test results are saved to `tests/output/`:
- Annotated images with bounding boxes
- Detection JSON files
- Performance metrics

---

## 6. Debugging

### 6.1 Enable Debug Logging

```bash
# Terminal
LOG_LEVEL=debug uvicorn main:app --reload

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 6.2 OpenVINO Profiling

```python
from openvino import Core

core = Core()
model = core.read_model("models/yolov8n/model.xml")
compiled = core.compile_model(model, "CPU")

# Enable performance counters
compiled.set_property({"PERF_COUNT": "YES"})

# After inference
request = compiled.create_infer_request()
request.infer(inputs)

# Get per-layer timing
perf_counts = request.profiling_info
for layer in perf_counts:
    print(f"{layer.node_name}: {layer.real_time.total_seconds() * 1000:.2f}ms")
```

### 6.3 Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: openvino` | Reinstall: `pip install openvino --upgrade` |
| Model load fails | Check `.xml` and `.bin` files exist in model directory |
| Slow first inference | Normal — OpenVINO compiles model on first run. Subsequent runs are faster. |
| Shape mismatch error | Verify input image is resized to model's expected dimensions |
| Out of memory | Use INT8 quantized model or reduce batch size |

### 6.4 Interactive Debugging

```bash
# Start with debugger support
python -m debugpy --listen 5678 --wait-for-client -m uvicorn main:app --reload
```

VS Code `launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach to FastAPI",
            "type": "python",
            "request": "attach",
            "connect": {"host": "localhost", "port": 5678}
        }
    ]
}
```

---

## 7. Code Style & Conventions

### 7.1 Formatting

```bash
# Format code with Black
black app/ tests/ --line-length 100

# Sort imports
isort app/ tests/

# Type checking
mypy app/ --ignore-missing-imports
```

### 7.2 Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Files | snake_case | `model_capabilities.py` |
| Classes | PascalCase | `InferenceService` |
| Functions | snake_case | `run_detection()` |
| Constants | UPPER_SNAKE | `DEFAULT_THRESHOLD` |
| API endpoints | kebab-case | `/inference/latest-frame` |

### 7.3 Documentation

- All public functions must have docstrings
- Use Google-style docstrings:

```python
def detect_objects(image: np.ndarray, threshold: float = 0.5) -> List[Detection]:
    """Detect objects in an image using the loaded model.
    
    Args:
        image: Input image as numpy array (H, W, C) in BGR format.
        threshold: Minimum confidence threshold for detections.
    
    Returns:
        List of Detection objects with class, confidence, and bbox.
    
    Raises:
        ModelNotLoadedError: If no model is currently loaded.
    """
```

---

## 8. Docker Development

### 8.1 Build Development Image

```bash
# Build with cache
docker build -t atriva-ai-openvino:dev .

# Build without cache (for clean builds)
docker build --no-cache -t atriva-ai-openvino:dev .
```

### 8.2 Run with Volume Mounts (for development)

```bash
docker run -d \
  -p 8001:8001 \
  -v $(pwd)/app:/app/app \
  -v $(pwd)/models:/app/models \
  -v /path/to/shared/frames:/shared/frames:ro \
  --name ai-dev \
  atriva-ai-openvino:dev
```

### 8.3 View Logs

```bash
# Follow logs
docker logs -f ai-dev

# Last 100 lines
docker logs --tail 100 ai-dev
```

### 8.4 Shell Access

```bash
docker exec -it ai-dev /bin/bash
```

---

## 9. Related Documentation

- ➡️ [API Endpoints](./api-endpints.md) — Complete REST API reference
- ➡️ [Pure Python Runtime](./pure-python.md) — Low-level OpenVINO usage
- ➡️ [Model Preparation](../models/preparation.md) — Converting models to OpenVINO
- ➡️ [Architecture](../architecture/README.md) — System design overview
- ➡️ [Testing Guide](../testing/README.md) — Running and writing tests
