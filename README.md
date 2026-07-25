# 🧠 NeuroScan AI — Brain Tumor Diagnostic Portal & API

NeuroScan AI is an end-to-end **Medical AI application** designed for automated **Brain Tumor Classification** from MRI scans. Built using **EfficientNetB0**, **FastAPI**, and **Docker**, it provides real-time diagnostic assistance with multi-class probability scores and an interactive web dashboard.

---

## 🌟 Key Features

- 🧠 **Multi-Class Classification**
  - Classifies MRI scans into **4 categories**:
    - `Glioma`
    - `Meningioma`
    - `No Tumor`
    - `Pituitary`

- 🤖 **Deep Learning Model**
  - Fine-tuned **EfficientNetB0** using **Transfer Learning**
  - Domain-specific data augmentation for improved accuracy

- ⚡ **RESTful API Backend**
  - High-performance asynchronous API built with **FastAPI**

- 🎨 **Interactive Web Dashboard**
  - Modern UI built using **Tailwind CSS**
  - Drag & Drop MRI upload
  - Instant image preview
  - Confidence score visualization
  - Responsive design

- 🐳 **Production-Ready Docker Support**
  - Fully containerized application
  - Run anywhere with Docker
  - Zero dependency installation

- 🔄 **Keras 3 Compatibility**
  - Supports both **Keras 3** and **tf_keras**
  - Reliable model deserialization across environments

---

# 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.9+ |
| **Deep Learning** | TensorFlow 2.16+, Keras, tf_keras |
| **Model** | EfficientNetB0 (Transfer Learning) |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | HTML5, Tailwind CSS, JavaScript (Fetch API) |
| **Containerization** | Docker Desktop / Docker Engine |
| **Image Processing** | Pillow (PIL), NumPy |

---

# 📂 Project Structure

```text
medical-ai-brain-tumor/
│
├── app/
│   └── main.py                  # FastAPI Application + Web Dashboard
│
├── models/
│   └── brain_tumor_model.h5     # Trained EfficientNetB0 Model
│
├── notebook/
│   └── model_training.ipynb     # Model Training Notebook
│
├── Dockerfile                   # Docker Configuration
├── requirements.txt             # Python Dependencies
└── README.md                    # Documentation
```

---

# 🚀 Quick Start (Docker)

## 📌 Prerequisites

- Install **Docker Desktop**
- Make sure Docker Engine is running

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Dhruv02-ux/medical-ai-brain-tumor.git

cd medical-ai-brain-tumor
```

---

## 2️⃣ Build Docker Image

```bash
docker build -t medical-ai-brain-tumor:v1 .
```

---

## 3️⃣ Run Docker Container

```bash
docker run -d \
-p 8000:8000 \
--name brain_tumor_service \
medical-ai-brain-tumor:v1
```

---

## 4️⃣ Open the Application

### 🌐 Web Dashboard

```
http://localhost:8000
```

### 📖 Swagger API Documentation

```
http://localhost:8000/docs
```

---

# 💡 API Usage

## Endpoint

```http
POST /predict
```

### Request

- **Content-Type**

```text
multipart/form-data
```

- Upload one MRI image

Supported formats:

- `.jpg`
- `.jpeg`
- `.png`

---

## Sample Response

```json
{
  "filename": "mri_scan.jpg",
  "diagnosis": "meningioma",
  "confidence_score": 0.877,
  "class_probabilities": {
    "glioma": 0.078,
    "meningioma": 0.877,
    "notumor": 0.028,
    "pituitary": 0.017
  }
}
```

---

# 📊 Prediction Classes

| Class | Description |
|--------|-------------|
| Glioma | Brain tumour originating from glial cells |
| Meningioma | Tumour arising from the meninges |
| No Tumor | Healthy MRI scan |
| Pituitary | Tumour affecting the pituitary gland |

---

# 🖼️ Web Dashboard Features

- 📤 Drag & Drop MRI Upload
- 🖼️ Live Image Preview
- 📈 Prediction Confidence Scores
- ⚡ Fast Inference
- 📱 Responsive Design
- 🌙 Clean Modern Interface

---

# 🧠 Deep Learning Pipeline

```text
MRI Image
     │
     ▼
Image Preprocessing
     │
     ▼
EfficientNetB0
(Transfer Learning)
     │
     ▼
Softmax Classification
     │
     ▼
Prediction + Confidence Scores
     │
     ▼
FastAPI Response
     │
     ▼
Web Dashboard
```

---

# ⚠️ Disclaimer

> **This project is intended for research, educational purposes, and screening assistance only.**
>
> It **must not** be used as a substitute for professional medical diagnosis or clinical decision-making by a certified radiologist or healthcare provider.

---

# 👨‍💻 Author

**Dhruv Dhiman**

- 🎓 B.Tech CSE (AI & Data Science)
- 💻 AI • Machine Learning • Deep Learning • Computer Vision

---

## ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.

Contributions, suggestions, and feedback are always welcome!
