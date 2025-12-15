
# Installing OpenVINO using openvino Python Environment 

This page covers installing OpenVINO from scratch, only if you would like to play with raw OpenVINO API.

---

# 1. Install OpenVINO (Python)

Use the official pip package:

```bash
python3 -m pip install --upgrade pip
pip install openvino openvino-dev
```

This installs:
- OpenVINO Runtime
- Model Optimizer
- INT8 calibration tools
- CLI utilities

# 2. Install OS Dependencies (Linux)

Ubuntu 22.04+ (CPU + GPU + NPU)

```bash
sudo apt install -y \
    build-essential \
    cmake \
    ocl-icd-opencl-dev \
    intel-opencl-icd \
    intel-level-zero-gpu level-zero \
    clinfo
```

Check GPU/NPU visibility:

```bash
clinfo | grep "Device"
```

# 3. Verify Your Installation

Run:
```python
from openvino.runtime import Core
ie = Core()
print(ie.available_devices)
```
Expected output:
```
['CPU']
['CPU', 'GPU']
['CPU', 'GPU', 'NPU']   # Meteor Lake AI Boost
```

# 4. Meteor Lake NPU Support

NPU requires:
- OpenVINO 2024.4+
- Linux kernel 6.8+
- Intel AI Boost driver (comes with Ubuntu 22.04 HWE or 24.04)
- Check NPU availability:

```python
print("NPU" in ie.available_devices)
```

# 5. Recommended Device Configurations
CPU (most stable)

```python
compiled = core.compile_model(model, "CPU", {
    "PERFORMANCE_HINT": "THROUGHPUT"
})
```

iGPU

```python
compiled = core.compile_model(model, "GPU")
```

NPU (Meteor Lake)

```python
compiled = core.compile_model(model, "NPU")
```

# 6. Verify Model Optimizer Availability

```bash
mo --help
```
If MO works, your install is correct.

Next Step

➡️ Model Preparation







