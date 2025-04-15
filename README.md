# 🎯 Lightweight Facial Segmentation Web App

This project implements a **CPU-friendly facial segmentation system** that identifies key facial components such as eyes, eyebrows, nose, mouth, and lips. It uses a custom lightweight U-Net model and provides an intuitive web interface built with **React (frontend)** and **Flask (backend)**.

---

## 📂 Dataset: CelebAMask-HQ

We use the **CelebAMask-HQ** dataset, which offers high-quality facial images and pixel-level annotations for various facial parts.

### 🔹 Dataset Structure

Download and extract dataset from "https://github.com/switchablenorms/CelebAMask-HQ"

## 🧠 Model Architectures

Two CPU-optimized PyTorch models are implemented:

### 1. `LightweightUNet`

- U-Net structure with fewer filters
- Maintains spatial precision with skip connections
- Suitable for facial segmentation on low-resource systems

### 2. `LightweightSegmentationModel`

- Uses **depthwise separable convolutions**
- Even fewer parameters than U-Net
- Ideal for fast CPU inference

---
## 🚀 How to Run the Fullstack App

You need to run both the backend and frontend from the `facial_segmentation_web/` directory.

### 🔧 1. Backend Setup (Flask)

```bash
cd facial_segmentation_web/backend
pip install -r requirements.txt
python app.py


```

### 🔧 1. Frontend Setup (React)

```bash
cd facial_segmentation_web/frontend
npm install
npm start
```


