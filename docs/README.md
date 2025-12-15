# Atriva AI API with OpenVINO 🚀

This is a FastAPI-based AI API that leverages **OpenVINO** for optimized deep learning inference.  
It provides a RESTful interface for running AI models, such as object detection and image classification.

## **⚡ Features**
✅ FastAPI-based AI API  
✅ OpenVINO optimization for inference  
✅ Dockerized for easy deployment  
✅ Comprehensive testing suite  
✅ Multiple AI models supported:
   - YOLOv8n Object Detection
   - LPRNet License Plate Recognition
   - Vehicle Detection and Tracking  

## **📂 Project Structure**

```plaintext
atriva-ai-openvino/
│── app/
│   ├── routes.py         # API route definitions
│   ├── services.py       # AI model processing logic
│   ├── models.py         # Data models and schemas
│   ├── model_capabilities.py  # Model capabilities and metadata
│   ├── shared_data.py    # Shared data utilities
│── models/               # Pretrained OpenVINO models
│   ├── yolov8n/          # YOLOv8n object detection model
│   ├── lprnet/           # LPRNet license plate recognition model
│   └── vehicle_tracking/ # Vehicle detection and tracking model
│── tests/                # Comprehensive testing suite
│   ├── test_runner.py    # Main test runner with model download/conversion
│   ├── test_yolov8_openvino.py   # YOLOv8 detection (supports n/s/m sizes)
│   ├── test_vehicle_tracking.py  # Vehicle tracking (IoU + ByteTrack)
│   ├── test_images/      # Sample test images
│   ├── test_videos/      # Sample test videos
│   ├── output/           # Generated output files
│   └── requirements.txt  # Test dependencies
│── main.py               # Entry point for FastAPI
│── config.py             # Configuration settings
│── requirements.txt      # Python dependencies
│── Dockerfile            # Docker configuration
│── README.md             # Project documentation
```
<!--
title: OpenVINO Overview
order: 11
-->

This section documents how to use the OpenVINO backend inside the Atriva AI Community Runtime.
OpenVINO provides highly optimized inference on Intel CPU, iGPU, and NPU (Meteor Lake/AI Boost).

## Structure

---

## **🚀 Getting Started**

### **Overview**

This section covers everything you need to get the Atriva AI API up and running. Whether you're setting up locally, using Docker, or looking for the quickest way to start, you'll find the instructions here. The setup process includes cloning the repository, installing dependencies, downloading AI models, and running the service.

### **🔧 Setup & Installation**

#### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/atriva-ai/atriva-ai-openvino.git
cd atriva-ai-openvino
```

#### **2️⃣ Create a Virtual Environment**
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

#### **3️⃣ Download AI Models**
```sh
# Download all required model files
cd tests
python test_runner.py --download-models

# Or download individual models
python test_runner.py --model yolov8n --download
python test_runner.py --model lprnet --download
python test_runner.py --model vehicle_tracking --download
```

**📝 Important**: Model binary files (.pt, .bin, .xml) are not included in the repository due to size constraints. They will be downloaded automatically when needed.

#### **4️⃣ Run the API Locally**
```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Access the API documentation at:  
👉 **http://localhost:8000/docs**

### **🐳 Running with Docker**

#### **1️⃣ Build the Docker Image**
```sh
docker build -t atriva-ai-openvino .
```

#### **2️⃣ Run the Container**
```sh
docker run -d -p 8000:8000 --name ai-openvino-container atriva-ai-openvino
```
Now, visit:  
👉 **http://localhost:8000/docs**

### **🚀 Quick Start (Recommended)**

**Models are pre-built and ready to use:**

```sh
# 1. Build Docker image (models already included)
docker build -t atriva-ai-openvino .

# 2. Run AI service
docker run -d -p 8001:8001 --name ai-inference atriva-ai-openvino

# 3. Test API
curl http://localhost:8001/models
```

