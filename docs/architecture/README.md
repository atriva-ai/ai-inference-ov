# OpenVINO AI Inference Service Architecture

This document describes the architecture of the Atriva AI Inference Service, a RESTful API backend that processes shared decoded frames from the video-pipeline container and returns inference results to the Atriva Core API.

---

## 1. System Overview

The Atriva AI Inference Service operates as a microservice within the Atriva ecosystem, bridging the video-pipeline and the Core API through AI-powered analysis.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Atriva Platform                                   │
│                                                                             │
│  ┌──────────────────┐    Shared Volume    ┌──────────────────────────────┐  │
│  │  Video-Pipeline  │ ──────────────────► │   AI Inference Service       │  │
│  │    Container     │   (Decoded Frames)  │      (OpenVINO)              │  │
│  │                  │                     │                              │  │
│  │  • RTSP/RTMP     │                     │  • FastAPI REST API          │  │
│  │  • Decode        │                     │  • OpenVINO Runtime          │  │
│  │  • Frame Export  │                     │  • Model Management          │  │
│  └──────────────────┘                     └──────────────┬───────────────┘  │
│                                                          │                  │
│                                                          │ Inference        │
│                                                          │ Results          │
│                                                          ▼                  │
│                                           ┌──────────────────────────────┐  │
│                                           │   Atriva Core API Backend    │  │
│                                           │                              │  │
│                                           │  • Event Processing          │  │
│                                           │  • Detection Storage         │  │
│                                           │  • Alert Management          │  │
│                                           └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components

### 2.1 FastAPI REST Server

The service exposes a RESTful API built with FastAPI, providing:

| Component | Purpose |
|-----------|---------|
| `main.py` | Application entry point, server configuration |
| `routes.py` | API endpoint definitions and request handling |
| `services.py` | Business logic and model inference orchestration |
| `models.py` | Pydantic schemas for request/response validation |

### 2.2 Shared Frame Access Layer

Frames decoded by the video-pipeline container are accessed through a shared volume:

```
/shared/frames/
├── camera1/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── latest.jpg
├── camera2/
│   └── ...
└── metadata/
    └── cameras.json
```

The `shared_data.py` module handles:
- Camera discovery and enumeration
- Frame file access and validation
- Latest frame retrieval per camera
- Frame metadata management

### 2.3 OpenVINO Inference Engine

The inference backend leverages Intel OpenVINO for optimized execution:

```
┌─────────────────────────────────────────────────────┐
│              OpenVINO Inference Engine              │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Model     │  │  Compiled   │  │   Infer     │  │
│  │   Loading   │──│   Model     │──│   Request   │  │
│  │   (.xml)    │  │   Cache     │  │   Queue     │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
│                                                     │
│  Supported Devices: CPU (INT8/FP16/FP32)            │
└─────────────────────────────────────────────────────┘
```

### 2.4 Model Registry

Pre-trained models optimized for OpenVINO:

| Model | Type | Use Case |
|-------|------|----------|
| YOLOv8n | Detection | General object detection |
| YOLOv8s/m | Detection | Higher accuracy detection |
| LPRNet | Recognition | License plate reading |
| Vehicle Tracking | Detection + Tracking | Vehicle analytics |

---

## 3. Data Flow

### 3.1 Frame Ingestion Flow

```
Video-Pipeline Container              AI Inference Service
        │                                     │
        │  1. Decode RTSP/RTMP Stream         │
        ▼                                     │
  ┌───────────┐                               │
  │  Decoded  │                               │
  │   Frame   │                               │
  └─────┬─────┘                               │
        │                                     │
        │  2. Write to Shared Volume          │
        ▼                                     │
  ┌───────────┐    3. Read Frame        ┌─────┴─────┐
  │  /shared/ │ ──────────────────────► │  Frame    │
  │  frames/  │                         │  Loader   │
  └───────────┘                         └─────┬─────┘
                                              │
                                              ▼
                                        ┌───────────┐
                                        │ Inference │
                                        │  Engine   │
                                        └───────────┘
```

### 3.2 Inference Request Flow

```
┌──────────────────┐
│  Atriva Core API │
│    or Client     │
└────────┬─────────┘
         │
         │  POST /inference/latest-frame
         │  POST /shared/cameras/{id}/inference
         ▼
┌──────────────────────────────────────────────────────┐
│                 AI Inference Service                 │
│                                                      │
│  1. Validate Request                                 │
│       │                                              │
│       ▼                                              │
│  2. Load Frame from Shared Volume                    │
│       │                                              │
│       ▼                                              │
│  3. Preprocess (Resize, Normalize, Tensor Convert)   │
│       │                                              │
│       ▼                                              │
│  4. Run OpenVINO Inference                           │
│       │                                              │
│       ▼                                              │
│  5. Post-process (NMS, Decode Boxes, Filter)         │
│       │                                              │
│       ▼                                              │
│  6. Return JSON Response                             │
│                                                      │
└──────────────────────────────────────────────────────┘
         │
         │  { "detections": [...], "camera_id": "..." }
         ▼
┌──────────────────┐
│  Atriva Core API │
└──────────────────┘
```

---

## 4. API Integration Points

### 4.1 Inbound APIs (Consumed by Core API)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health and shared volume status |
| `/models` | GET | Available models list |
| `/shared/cameras` | GET | Cameras with available frames |
| `/shared/cameras/{id}/inference` | POST | Run inference on camera's latest frame |
| `/inference/latest-frame` | POST | Inference on specified camera frame |
| `/inference/background` | POST | Start background inference worker |

### 4.2 Outbound Communication (To Core API)

Inference results are returned synchronously via HTTP response:

```json
{
  "camera_id": "camera1",
  "frame_path": "/shared/frames/camera1/latest.jpg",
  "timestamp": "2025-12-14T10:30:00Z",
  "model_name": "yolov8n",
  "detections": [
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.92,
      "bbox": [150, 200, 350, 450]
    }
  ]
}
```

---

## 5. Deployment Architecture

### 5.1 Container Topology

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Host                            │
│                                                             │
│  ┌─────────────────┐         ┌─────────────────────────┐    │
│  │ video-pipeline  │         │   ai-inference          │    │
│  │                 │         │                         │    │
│  │  Port: N/A      │         │  Port: 8001:8001        │    │
│  │                 │         │                         │    │
│  │  Volumes:       │         │  Volumes:               │    │
│  │  - /shared      │◄───────►│  - /shared (readonly)   │    │
│  │                 │         │  - /models              │    │
│  └─────────────────┘         └─────────────────────────┘    │
│                                       │                     │
│                                       │ REST API            │
│                                       ▼                     │
│                              ┌─────────────────────────┐    │
│                              │   atriva-core-api       │    │
│                              │                         │    │
│                              │  Port: 8080:8080        │    │
│                              └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Volume Mounts

| Volume | Purpose | Access |
|--------|---------|--------|
| `/shared/frames` | Decoded video frames | Read-only |
| `/models` | OpenVINO model files | Read-only |
| `/tmp/inference` | Temporary processing | Read-write |

---

## 6. Performance Considerations

### 6.1 Accelerator Selection

| Accelerator | Precision | Use Case |
|-------------|-----------|----------|
| `cpui8` | INT8 | Maximum throughput, slight accuracy trade-off |
| `cpu16` | FP16 | Balanced performance/accuracy |
| `cpu32` | FP32 | Maximum accuracy, baseline performance |

### 6.2 Optimization Strategies

- **Model Caching**: Compiled models are cached to avoid recompilation
- **Frame Polling**: Efficient file-based frame access from shared volume
- **Async Inference**: Non-blocking inference for high-throughput scenarios
- **Batch Processing**: Multiple frames processed in single inference call (when supported)

---

## 7. Security Considerations

- Shared volume mounted as read-only for frame access
- API endpoints validated with Pydantic schemas
- No direct external network access required
- Internal service communication only

---

## Next Steps

- ➡️ [API Endpoints](../development/api-endpints.md) — Detailed endpoint documentation
- ➡️ [Models](../models/README.md) — Supported models and preparation
- ➡️ [Development](../development/README.md) — Local development setup
