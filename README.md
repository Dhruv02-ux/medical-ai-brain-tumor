# 🧠 NeuroScan AI — Brain Tumor Diagnostic Portal & API
 
NeuroScan AI is an end-to-end Medical AI application designed for automated Brain Tumor Classification from MRI scans. Built using EfficientNetB0, FastAPI, and Docker, it provides real-time diagnostic assistance with multi-class probability scores and an interactive web dashboard.
 
## 🌟 Key Features
 
- **Multi-Class Classification**: Classifies MRI scans into 4 distinct categories:
  - `Glioma`
  - `Meningioma`
  - `No Tumor`
  - `Pituitary`
- **Deep Learning Model**: Fine-tuned EfficientNetB0 architecture trained with transfer learning and domain-specific data augmentation.
- **RESTful API Backend**: High-performance asynchronous endpoint built using FastAPI.
- **Embedded Web UI Dashboard**: Built-in interactive Web UI (Tailwind CSS) with drag-and-drop MRI upload, instant preview, and confidence score breakdown.
- **Production-Ready Containerization**: Fully containerized using Docker for zero-dependency execution across local and cloud environments.
- **Keras 3 / tf_keras Compatibility**: Robust model deserialization handling for seamless cross-framework execution.
## 🛠️ Tech Stack & Tools
 
- **Core Language**: Python 3.9+
- **Deep Learning**: TensorFlow 2.16+, Keras, `tf_keras`
- **Model Architecture**: EfficientNetB0 (Transfer Learning)
- **Backend Framework**: FastAPI, Uvicorn
- **Frontend UI**: HTML5, Tailwind CSS, JavaScript (Fetch API)
- **Containerization**: Docker Desktop / Docker Engine
- **Image Processing**: Pillow (PIL), NumPy
## 📐 Project Architecture
 
```
medical-ai-brain-tumor/
├── app/
│   └── main.py                  # FastAPI Application + UI Dashboard
├── models/
│   └── brain_tumor_model.h5     # Fine-tuned Trained AI Model
├── notebook/
│   └── model_training.ipynb     # Model Training & Augmentation Notebook
├── Dockerfile                   # Docker Image Configuration
├── requirements.txt             # Project Dependencies
└── README.md                    # Project Documentation
```
 
## 🚀 Quickstart Guide (Running via Docker)
 
### Prerequisites
 
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
### 1. Clone the Repository
 
```bash
git clone https://github.com/Dhruv02-ux/medical-ai-brain-tumor.git
cd medical-ai-brain-tumor
```
 
### 2. Build the Docker Image
 
```bash
docker build -t medical-ai-brain-tumor:v1 .
```
 
### 3. Run the Container
 
```bash
docker run -d -p 8000:8000 --name brain_tumor_service medical-ai-brain-tumor:v1
```
 
### 4. Open in Browser
 
- Interactive UI Portal: [http://localhost:8000](http://localhost:8000)
- Interactive Swagger API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
## 💡 API Usage
 
**Endpoint:** `POST /predict`
 
- **Request Body**: `multipart/form-data` with an MRI image file (`.jpg`, `.jpeg`, `.png`).
- **Sample Response**:
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
 
## ⚠️ Disclaimer
 
This system is intended for research and screening assistance only. It should not replace professional clinical evaluation by a certified radiologist.
