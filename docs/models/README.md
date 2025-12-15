
# Model Preparation for OpenVINO

This section describes how to convert ONNX models into OpenVINO IR format, quantize for CPU/NPU, and validate compatibility.

OpenVINO IR is required for:
- Optimal CPU performance
- iGPU/NPU execution
- INT8 hardware acceleration


## **🏗️ Model Build Architecture**

The AI service uses a **pre-built model approach** for optimal performance:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Development   │    │   Docker Build   │    │   Production    │
│   (Host)        │    │   (Container)    │    │   (Runtime)     │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Scripts       │    │ • Copy models    │    │ • Load models   │
│ • Model conv.   │───▶│ • Install deps   │───▶│ • Run inference │
│ • Testing       │    │ • Fast build     │    │ • API endpoints │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 1. Export Model to ONNX

### PyTorch example:
```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"]
)
```

TensorFlow:
```bash
python -m tf2onnx.convert --saved-model ./saved --output model.onnx
```

## 2. Convert ONNX → OpenVINO IR

Use Model Optimizer:

```bash
mo --input_model model.onnx --output_dir ./ir --compress_to_fp16
```

Produces:

- model.xml
- model.bin


FP16 recommended for:
- CPU
- iGPU
- NPU

For FP32:

```bash
mo --input_model model.onnx --data_type FP32
```

## 3. Quantization (INT8)

Use POT (Post-Training Optimization Tool):

```python
from openvino.tools.pot import DataLoader, IEEngine, load_model, save_model
```

CLI version:
```bash
pot -m model.xml -d config.json -o output/
```

Benefits:
- Faster CPU execution
- Required for optimal NPU performance

## 4. Validate the IR Model

```python
from openvino.runtime import Core

ie = Core()
model = ie.read_model("model.xml")
compiled = ie.compile_model(model, "CPU")
```

## 5. Check Input/Output Shapes

```python
for inp in model.inputs:
    print(inp.get_any_name(), inp.shape, inp.element_type)
```

Ensure:
- NCHW or NHWC matches your preprocessing
- Dynamic shapes are allowed (-1)

## 6. Layout Conversion

If your model expects NHWC but your pipeline is NCHW:

```bash
mo --input_model model.onnx --layout=input(NHWC)
```

## 7. Batch Size

Static batch:
```bash
mo --input_model model.onnx --batch 4
```

Dynamic batch:
```bash
mo --input_model model.onnx --input_shape "[?,3,224,224]"
```

## 8. Best Practices

Prefer FP16 for GPU/NPU
Prefer INT8 for CPU (if accuracy drop acceptable)
Always validate the model with a real input
Use dynamic input shapes for video analytics pipelines

Next Step

➡️ Continue to Runtime API

