import numpy as np
import cv2
# from openvino import AsyncInferQueue
from openvino.runtime import InferRequest
from app.models import model_manager

def letterbox(img, new_shape=(640,640), color=(114,114,114)):
    """Letterbox resizing to preserve aspect ratio"""
    h0, w0 = img.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w0, new_h / h0)
    resized_w, resized_h = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (resized_w, resized_h))
    dw = new_w - resized_w
    dh = new_h - resized_h
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)

def preprocess_image(image_bytes: bytes, input_shape: tuple, use_letterbox: bool = True):
    """Preprocess image for inference
    
    Args:
        image_bytes: Raw image bytes
        input_shape: Model input shape (N, C, H, W)
        use_letterbox: If True, use letterbox (YOLOv8). If False, use direct resize (SSD models)
    
    Returns:
        Tuple of (input_data, original_shape, ratio, pad)
    """
    _, _, h, w = input_shape  # Extract height and width from model shape
    target_shape = (w, h)  # Model expects (width, height)

    # Convert bytes to NumPy array
    image_array = np.frombuffer(image_bytes, np.uint8)

    # Decode image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Store original shape for postprocessing
    original_shape = image.shape

    if use_letterbox:
        # Use letterbox resizing to preserve aspect ratio (YOLOv8)
        resized_image, ratio, pad = letterbox(image, new_shape=target_shape)
    else:
        # Direct resize without letterbox (SSD models)
        resized_image = cv2.resize(image, target_shape)
        ratio = 1.0  # Will use normalized coords
        pad = (0, 0)
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    normalized_image = rgb_image.astype(np.float32) / 255.0
    
    # Transpose to CHW format
    input_data = np.transpose(normalized_image, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)
    
    print(f"Preprocessed input shape: {input_data.shape}")
    print(f"Original image shape: {original_shape}")
    print(f"Letterbox: {use_letterbox}, ratio: {ratio}, pad: {pad}")

    return input_data, original_shape, ratio, pad

def run_inference(input_data, compiled_model):
    """Runs inference on input data using OpenVINO model."""
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Debug: Check input shape before inference
    print(f"Input shape before inference: {input_data.shape}, expected: {input_layer.shape}")

    # Convert input to OpenVINO-compatible format
    input_tensor = np.array(input_data, dtype=np.float32)

    # Create inference request
    infer_request = compiled_model.create_infer_request()

    # Start inference synchronously
    infer_request.infer({input_layer: input_tensor})
    print("Inference completed...")

    # Get the output - YOLOv8 format: (1, 84, 8400) for NMS output
    output = infer_request.get_output_tensor(0).data
    print(f"Output shape after inference: {output.shape}")

    return output

def postprocess_detections(outputs, original_shape, ratio, pad, classes, confidence_threshold=0.25, nms_threshold=0.45, model_type="yolov8"):
    """Postprocess detections with NMS and proper coordinate transformation
    
    Supports multiple output formats:
    - YOLOv8 raw: (1, 84, 8400) - center format with class scores
    - YOLOv8 NMS: (1, 6, N) - corner format with [x1, y1, x2, y2, score, class_id]
    - SSD/OpenVINO: (1, 1, N, 7) - normalized coords [image_id, label, conf, xmin, ymin, xmax, ymax]
    """
    detections = []
    
    # Get the output tensor
    if isinstance(outputs, dict):
        output_tensor = list(outputs.values())[0]
    else:
        output_tensor = outputs
    
    print(f"Processing output shape: {output_tensor.shape}, model_type: {model_type}")
    
    orig_h, orig_w = original_shape[:2]
    
    # Detect output format based on shape
    if len(output_tensor.shape) == 4 and output_tensor.shape[3] == 7:
        # SSD/OpenVINO Model Zoo format: (1, 1, N, 7)
        # Each detection: [image_id, label, confidence, x_min, y_min, x_max, y_max]
        # Coordinates are normalized [0, 1]
        print("Processing SSD/OpenVINO output format")
        
        for detection in output_tensor[0, 0]:
            image_id, label, confidence, x_min, y_min, x_max, y_max = detection
            
            if confidence > confidence_threshold:
                # Convert normalized coords to pixel coords
                x1 = int(x_min * orig_w)
                y1 = int(y_min * orig_h)
                x2 = int(x_max * orig_w)
                y2 = int(y_max * orig_h)
                
                # Clip to image boundaries
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))
                
                # Get class name (label is 1-indexed for some models)
                class_id = int(label)
                if class_id < len(classes):
                    class_name = classes[class_id]
                elif class_id - 1 < len(classes) and class_id > 0:
                    # Try 0-indexed
                    class_name = classes[class_id - 1]
                    class_id = class_id - 1
                else:
                    class_name = f"class_{class_id}"
                
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "bbox_xyxy": [x1, y1, x2, y2]
                })
    
    elif len(output_tensor.shape) == 3 and output_tensor.shape[1] == 6:
        # YOLOv8 NMS output format: (1, 6, N) where each detection has [x1, y1, x2, y2, score, class_id]
        print("Processing YOLOv8 NMS output format")
        for i in range(output_tensor.shape[2]):
            x1, y1, x2, y2, score, class_id = output_tensor[0, :, i]
            
            if score > confidence_threshold:
                # Undo letterbox transformation
                x1 = (x1 - pad[0]) / ratio
                y1 = (y1 - pad[1]) / ratio
                x2 = (x2 - pad[0]) / ratio
                y2 = (y2 - pad[1]) / ratio
                
                # Clip to image boundaries
                x1 = max(0, min(int(x1), orig_w - 1))
                y1 = max(0, min(int(y1), orig_h - 1))
                x2 = max(0, min(int(x2), orig_w - 1))
                y2 = max(0, min(int(y2), orig_h - 1))
                
                # Get class name
                if int(class_id) < len(classes):
                    class_name = classes[int(class_id)]
                else:
                    class_name = f"class_{int(class_id)}"
                
                detections.append({
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(score),
                    "bbox_xyxy": [x1, y1, x2, y2]
                })
    
    elif len(output_tensor.shape) == 3:
        # Raw YOLOv8 format: (1, 84, 8400) - need to apply NMS
        print("Processing raw YOLOv8 output format with NMS")
        
        # Transpose to (8400, 84) for easier processing
        predictions = output_tensor[0].T
        
        boxes = []
        confidences = []
        class_ids = []
        
        for pred in predictions:
            # First 4 values: cx, cy, w, h
            cx, cy, w, h = pred[:4]
            # Remaining values: class scores
            class_scores = pred[4:]
            
            # Get best class
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > confidence_threshold:
                # Convert from center format to corner format
                x1 = cx - w / 2
                y1 = cy - h / 2
                
                # Undo letterbox transformation
                x1_orig = (x1 - pad[0]) / ratio
                y1_orig = (y1 - pad[1]) / ratio
                w_orig = w / ratio
                h_orig = h / ratio
                
                # Store as [x, y, w, h] for NMS
                boxes.append([int(x1_orig), int(y1_orig), int(w_orig), int(h_orig)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
        
        # Apply Non-Maximum Suppression
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            
            print(f"Before NMS: {len(boxes)}, After NMS: {len(indices) if len(indices) > 0 else 0}")
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    
                    # Clip to image boundaries
                    x1 = max(0, min(x, orig_w - 1))
                    y1 = max(0, min(y, orig_h - 1))
                    x2 = max(0, min(x + w, orig_w - 1))
                    y2 = max(0, min(y + h, orig_h - 1))
                    
                    # Get class name
                    if class_id < len(classes):
                        class_name = classes[class_id]
                    else:
                        class_name = f"class_{class_id}"
                    
                    detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidences[i],
                        "bbox_xyxy": [x1, y1, x2, y2]
                    })
    
    else:
        print(f"Unknown output format: {output_tensor.shape}")
    
    print(f"Found {len(detections)} detections")
    return detections

def run_object_detection(image_bytes: bytes, object_name: str):
    """Runs inference using a dynamically selected model (YOLOv8 or SSD)."""
    compiled_model, input_shape = model_manager.load_model(object_name)
    
    # Get model configuration for classes and thresholds
    model_config = model_manager.get_model_config(object_name)
    classes = model_config.get("classes", [])
    confidence_threshold = model_config.get("confidence_threshold", 0.25)
    nms_threshold = model_config.get("nms_threshold", 0.45)
    model_type = model_config.get("model_type", "object_detection")
    
    # Determine preprocessing based on model type
    # YOLOv8 uses letterbox, SSD/OpenVINO models use direct resize
    use_letterbox = "yolo" in object_name.lower() or model_type == "object_detection"
    if "face-detection" in object_name or "license-plate" in object_name or "barrier" in object_name:
        use_letterbox = False
    
    print(f"Model: {object_name}, Type: {model_type}, Input shape: {input_shape}, Letterbox: {use_letterbox}")
    
    # Preprocess image for model input
    image_data, original_shape, ratio, pad = preprocess_image(image_bytes, input_shape, use_letterbox)

    # Run inference
    model_output = run_inference(image_data, compiled_model)

    # Postprocess detections with proper coordinate transformation
    detections = postprocess_detections(
        model_output, 
        original_shape, 
        ratio, 
        pad, 
        classes, 
        confidence_threshold,
        nms_threshold,
        model_type
    )

    # Print results
    for det in detections:
        print(f"Class: {det['class_name']} (ID: {det['class_id']}), Confidence: {det['confidence']:.3f}, BBox: {det['bbox_xyxy']}")

    return detections


def draw_detections_on_image(image_bytes: bytes, detections: list) -> bytes:
    """Draw bounding boxes and labels on image based on detections.
    
    Args:
        image_bytes: Raw image bytes
        detections: List of detection dictionaries with 'bbox_xyxy', 'class_name', 'confidence'
    
    Returns:
        Annotated image as bytes
    """
    import cv2
    import numpy as np
    
    # Convert bytes to NumPy array
    image_array = np.frombuffer(image_bytes, np.uint8)
    # Decode image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    annotated_image = image.copy()
    
    # Color palette for different classes
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    for detection in detections:
        if 'bbox_xyxy' not in detection:
            continue
            
        x1, y1, x2, y2 = detection['bbox_xyxy']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        class_name = detection.get('class_name', 'unknown')
        confidence = detection.get('confidence', 0.0)
        class_id = detection.get('class_id', 0)
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_image, (x1, y1 - label_h - 10), 
                     (x1 + label_w, y1), color, -1)
        cv2.putText(annotated_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Encode annotated image back to bytes
    _, encoded_image = cv2.imencode('.jpg', annotated_image)
    return encoded_image.tobytes()

def recognize_text_openvino(image_bytes: bytes) -> dict:
    """
    Recognize text using OpenVINO text-recognition-0012 model.
    Used for license plate recognition in Docker API (no EasyOCR dependency).
    
    Args:
        image_bytes: Raw image bytes (cropped text region)
    
    Returns:
        dict with 'text' and 'confidence'
    """
    try:
        # Load text recognition model
        compiled_model, input_shape = model_manager.load_model("text-recognition-0012")
        
        # Decode image
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"text": "", "confidence": 0, "error": "Failed to decode image"}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input: [1, 32, 120, 1] -> (120, 32)
        resized = cv2.resize(gray, (120, 32))
        
        # Normalize and reshape
        normalized = resized.astype(np.float32) / 255.0
        input_data = normalized.reshape(1, 32, 120, 1)
        
        # Run inference
        result = compiled_model(input_data)
        outputs = result[compiled_model.output(0)]
        
        # Decode CTC output [30, 1, 37]
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        predictions = outputs[:, 0, :]  # Shape: [30, 37]
        
        # Apply softmax
        exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        
        indices = np.argmax(probs, axis=1)
        
        # CTC decode: remove blanks and repeated characters
        text = ""
        conf_sum = 0
        conf_count = 0
        prev_idx = -1
        
        for i, idx in enumerate(indices):
            if idx != 36 and idx != prev_idx:  # 36 is blank
                if idx < len(chars) - 1:
                    text += chars[idx]
                    conf_sum += probs[i, idx]
                    conf_count += 1
            prev_idx = idx
        
        avg_conf = conf_sum / conf_count if conf_count > 0 else 0
        
        return {
            "text": text,
            "confidence": float(avg_conf),
            "model": "text-recognition-0012"
        }
        
    except Exception as e:
        return {"text": "", "confidence": 0, "error": str(e)}