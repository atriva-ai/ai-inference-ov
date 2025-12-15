# Atriva AI API with OpenVINO Docker Guide 🚀

This page covers installing OpenVINO in Docker environment, which is how Atriva AI services deployed.

## **🐳 Running with Docker**

### **1️⃣ Build the Docker Image**
```sh
docker build -t atriva-ai-openvino .
```

### **2️⃣ Run the Container**
```sh
docker run -d -p 8000:8000 --name ai-openvino-container atriva-ai-openvino
```
Now, visit:  
👉 **http://localhost:8000/docs**
