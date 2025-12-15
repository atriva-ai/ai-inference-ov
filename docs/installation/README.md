# Atriva AI API with OpenVINO Installation Guide 🚀

This page covers installing OpenVINO and preparing your system for Atriva API Service.

## **🔧 Setup & Installation**

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/atriva-ai/atriva-ai-openvino.git
cd atriva-ai-openvino
```

### **2️⃣ Create a Virtual Environment**
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### **3️⃣ Download AI Models**
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

### **4️⃣ Run the API Locally**
```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Access the API documentation at:  
👉 **http://localhost:8000/docs**