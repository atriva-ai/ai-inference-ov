import os
import shutil
import requests
import zipfile
import json
from openvino import Core
from config import MODEL_DIR, MODEL_URLS

# Define OpenVINO-compatible accelerators
ACCELERATORS = ["cpui8", "cpu16", "cpu32"]

# Mapping user-friendly model names to actual OpenVINO models
MODEL_NAME_MAPPING = {
    # YOLOv8 models (80 COCO classes)
    "yolov8n": "yolov8n",
    "yolov8s": "yolov8s",
    "yolov8m": "yolov8m",
    # Convenience aliases for common detections (use YOLOv8n)
    "vehicle": "yolov8n",
    "car": "yolov8n",
    "person": "yolov8n",
    "object": "yolov8n",
    # Face detection (OpenVINO Model Zoo)
    "face-detection-retail-0005": "face-detection-retail-0005",
    "face": "face-detection-retail-0005",
    # Vehicle & License Plate Detection (OpenVINO Model Zoo)
    "vehicle-license-plate-detection-barrier-0106": "vehicle-license-plate-detection-barrier-0106",
    "license_plate_detection": "vehicle-license-plate-detection-barrier-0106",
    # Text recognition (Western alphanumeric)
    "text-recognition-0012": "text-recognition-0012",
    "text_recognition": "text-recognition-0012",
    "ocr": "text-recognition-0012",
    # License plate recognition (Chinese - legacy)
    "lprnet": "lprnet",
    "license_plate": "lprnet",
    # Person re-identification (for tracking)
    "person-reidentification-retail-0286": "person-reidentification-retail-0286",
    "person_reid": "person-reidentification-retail-0286",
    "reid": "person-reidentification-retail-0286"
}

# OpenVINO Inference Engine Initialization
class ModelManager:
    BASE_DIR = MODEL_DIR

    def __init__(self, acceleration="cpu32"):
        if acceleration not in ACCELERATORS:
            raise ValueError(f"❌ Unsupported accelerator: {acceleration}")
        
        self.ie = Core()  # OpenVINO Inference Engine
        # Debug: Print available devices
        devices = self.ie.available_devices
        print(f"Available OpenVINO devices: {devices}")

        self.acceleration = acceleration
        self.MODEL_DIR = os.path.join(self.BASE_DIR, acceleration)
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    # def download_model(self, model_name):
    #     """Download model files (XML & BIN) and save them properly."""
    #     # DISABLED: Automatic download is disabled. Model files must be committed to repository.
    #     raise Exception(f"❌ Automatic download is disabled. Please ensure model files are committed to the repository.")

    def load_model(self, requested_model):
        """Load a model using its friendly name or actual name."""
        model_name = MODEL_NAME_MAPPING.get(requested_model, requested_model)
        if model_name not in MODEL_URLS:
            raise ValueError(f"❌ Unknown model: {requested_model}. Available: {list(MODEL_NAME_MAPPING.keys())}")

        # All models are loaded from models/{model_name}/ directory
        model_folder = os.path.join(self.BASE_DIR, model_name)
        xml_path = os.path.join(model_folder, f"{model_name}.xml")
        bin_path = os.path.join(model_folder, f"{model_name}.bin")

        # Check if the model exists
        if os.path.exists(xml_path) and os.path.exists(bin_path):
            print(f"✅ Model {model_name} found locally in {model_folder}")
            
            # Load the network model (XML & BIN)
            model = self.ie.read_model(model=xml_path)

            # Compile the model for inference
            compiled_model = self.ie.compile_model(model=model, device_name="CPU")
            print(f"✅ Model {model_name} compiled successfully on device: CPU")
            
            input_shape = compiled_model.input(0).shape  # Expected shape (N, C, H, W)
            return compiled_model, input_shape

        print(f"❌ Model {model_name} not found locally in {model_folder}")
        print(f"❌ Expected files: {xml_path}, {bin_path}")
        raise Exception(f"❌ Model {model_name} files not found. Please ensure model files are committed to the repository.")

    def get_model_config(self, requested_model):
        """Get model configuration including classes and thresholds."""
        model_name = MODEL_NAME_MAPPING.get(requested_model, requested_model)
        
        # Try to load configuration from model.json
        config_path = os.path.join(self.BASE_DIR, model_name, "model.json")
        if os.path.exists(config_path):
            print(f"✅ Loading config from {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        
        # Fallback: try to get from model_capabilities
        from app.model_capabilities import MODEL_CAPABILITIES
        if model_name in MODEL_CAPABILITIES:
            caps = MODEL_CAPABILITIES[model_name]
            return {
                "model_type": caps.get("type", "object_detection"),
                "classes": caps.get("capabilities", {}).get("detection_classes", ["object"]),
                "confidence_threshold": caps.get("capabilities", {}).get("confidence_threshold", 0.5),
                "nms_threshold": caps.get("capabilities", {}).get("nms_threshold", 0.4)
            }
        
        # Default fallback
        return {
            "classes": ["object"],
            "confidence_threshold": 0.3,
            "nms_threshold": 0.5
        }

    def list_models(self):
        """List all available models."""
        available_models = []
        
        # Check for model directories with model.json config files
        if os.path.exists(self.BASE_DIR):
            for model_name in os.listdir(self.BASE_DIR):
                model_dir = os.path.join(self.BASE_DIR, model_name)
                if os.path.isdir(model_dir):
                    # Check if it has model files (xml/bin) or config
                    xml_path = os.path.join(model_dir, f"{model_name}.xml")
                    config_path = os.path.join(model_dir, "model.json")
                    
                    if os.path.exists(xml_path) or os.path.exists(config_path):
                        available_models.append(model_name)
        
        return available_models

# Global model manager instance
model_manager = ModelManager(acceleration="cpu32")