# 🧰 Atriva AI OpenVINO Inference API — Troubleshooting Guide

This document provides common troubleshooting steps for issues encountered when working with the Atriva AI Runtime using OpenVINO.  
Use this guide to quickly resolve Python environment problems, model conversion issues, and runtime errors.

- **Covers:**  
  - Python and dependency errors  
  - Model download and conversion failures  
  - API and environment quirks

> ℹ️ **Tip:** If you do not see your problem listed, check the latest README and development guides, or open an [issue on GitHub](https://github.com/atriva-ai/atriva-inference/issues).

---

## **🚨 Troubleshooting**

### **Overview**

The Troubleshooting section provides solutions to common issues you may encounter when setting up or using the Atriva AI API. This includes Python version compatibility problems, model download and conversion issues, and API usage errors. Each issue includes a description of the problem and step-by-step solutions.

### **Python Version Issues**

#### **Error: "No matching distribution found for torch"**
```bash
# Problem: Using Python 3.13
# Solution: Use Python 3.11
pyenv local 3.11.13
python3.11 -m venv venv-py311
source venv-py311/bin/activate
pip install -r requirements.txt
```

#### **Error: "NumPy compatibility issues"**
```bash
# Problem: NumPy 2.x with PyTorch
# Solution: Downgrade NumPy
pip install "numpy<2"
```

#### **Error: "ModuleNotFoundError: No module named 'ultralytics'"**
```bash
# Problem: Missing PyTorch dependencies
# Solution: Use Python 3.11 environment
cd scripts
pyenv local 3.11.13
python3.11 -m venv scripts-venv-py311
source scripts-venv-py311/bin/activate
pip install -r requirements.txt
```

### **Model Issues**

#### **Error: "Unable to read the model: model.xml"**
```bash
# Problem: Corrupted model files (HTML error page instead of XML)
# Solution: Re-download using test_runner.py which handles conversion properly
cd tests
python test_runner.py --download-models
```

#### **Error: "404 Not Found" when downloading models**
```bash
# Problem: Direct OpenVINO model URLs don't exist for YOLOv8
# Solution: Use ultralytics to download PyTorch weights and convert to OpenVINO
# test_runner.py handles this automatically:
python test_runner.py --download-models

# Manual conversion:
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically
model.export(format='openvino')  # Converts to .xml/.bin
"
```

#### **Error: "Wrong class labels (class_34 instead of car)"**
```bash
# Problem: Loading classes from wrong metadata.yaml file
# Solution: Ensure ultralytics/metadata.yaml is present (has 80 COCO classes)
# The test scripts prioritize loading from models/yolov8n/ultralytics/metadata.yaml
```

#### **Error: "Incompatible inputs of type: ConstOutput"**
```bash
# Problem: Incorrect OpenVINO inference API usage
# Old incorrect code: result = model([input_tensor], {input_tensor: input_data})
# Correct code: result = model(input_data)
```

#### **Error: "too many values to unpack (expected 6)"**
```bash
# Problem: YOLOv8 output format is (1, 84, 8400) not (1, N, 6)
# Solution: Transpose output and parse correctly:
# predictions = outputs[0].T  # Shape: (8400, 84)
# First 4 values: cx, cy, w, h
# Remaining 80 values: class scores
```

### **Model File Structure**
Each model directory contains:
- `model.json` - Model configuration and metadata
- `metadata.yaml` - Additional model information
- `*.xml` - OpenVINO model structure (downloaded)
- `*.bin` - OpenVINO model weights (downloaded)
- `*.pt` - PyTorch model weights (optional, for conversion)

**Note**: Binary files (.pt, .bin, .xml) are not stored in Git due to size constraints. They are downloaded automatically when needed.