import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
try:
    import tf_keras as keras
except ImportError:
    import tensorflow.keras as keras

from PIL import Image
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

try:
    import keras as keras_raw
    orig_layer_init = keras_raw.layers.Layer.__init__
    def patched_layer_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        orig_layer_init(self, *args, **kwargs)
    keras_raw.layers.Layer.__init__ = patched_layer_init
except Exception:
    pass

try:
    orig_tf_layer_init = tf.keras.layers.Layer.__init__
    def patched_tf_layer_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        orig_tf_layer_init(self, *args, **kwargs)
    tf.keras.layers.Layer.__init__ = patched_tf_layer_init
except Exception:
    pass

app = FastAPI(
    title="Medical AI - Brain Tumor Diagnostic API",
    description="Deep Learning API for Brain Tumor MRI Classification"
)

MODEL_PATH = "models/brain_tumor_model.h5"
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

model = None
model_load_error = None

try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(">>> Model loaded successfully! <<<", flush=True)
except Exception as e:
    model_load_error = str(e)
    print(f">>> Error loading model: {e} <<<", flush=True)

def preprocess_mri(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    if hasattr(keras.applications, 'efficientnet'):
        img_array = keras.applications.efficientnet.preprocess_input(img_array)
    else:
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

@app.get("/", response_class=HTMLResponse)
def render_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NeuroScan AI - Brain Tumor Diagnostic Portal</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    </head>
    <body class="bg-slate-900 text-slate-100 min-h-screen font-sans">
        <div class="max-w-6xl mx-auto px-4 py-8">
            <!-- Header -->
            <header class="flex flex-col md:flex-row justify-between items-center pb-8 border-b border-slate-800 mb-8">
                <div class="flex items-center space-x-3 mb-4 md:mb-0">
                    <div class="bg-indigo-600 p-3 rounded-xl shadow-lg shadow-indigo-500/30">
                        <i class="fa-solid fa-brain text-2xl text-white"></i>
                    </div>
                    <div>
                        <h1 class="text-2xl font-bold text-white tracking-wide">NeuroScan AI</h1>
                        <p class="text-xs text-slate-400">Deep Learning MRI Diagnostic Portal</p>
                    </div>
                </div>
                <div class="flex items-center space-x-2 bg-emerald-500/10 border border-emerald-500/30 px-4 py-1.5 rounded-full">
                    <span class="w-2.5 h-2.5 bg-emerald-400 rounded-full animate-pulse"></span>
                    <span class="text-xs font-semibold text-emerald-400">AI Model Active & Ready</span>
                </div>
            </header>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <!-- Left Panel: Upload Area -->
                <div class="bg-slate-800/50 border border-slate-700/60 rounded-2xl p-6 backdrop-blur">
                    <h2 class="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
                        <i class="fa-solid fa-cloud-arrow-up text-indigo-400"></i> Upload MRI Scan
                    </h2>
                    
                    <div id="drop-area" onclick="document.getElementById('mri-input').click()" 
                         class="border-2 border-dashed border-slate-600 hover:border-indigo-500 rounded-xl p-8 text-center cursor-pointer transition-all duration-300 bg-slate-900/40 hover:bg-indigo-950/20 group">
                        <input type="file" id="mri-input" accept="image/jpeg,image/png,image/jpg" class="hidden" onchange="handleFileSelect(event)">
                        
                        <div id="upload-prompt" class="space-y-3">
                            <div class="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto group-hover:scale-110 transition-transform">
                                <i class="fa-solid fa-file-medical text-2xl text-indigo-400"></i>
                            </div>
                            <p class="text-sm font-medium text-slate-300">Click or Drag & Drop MRI Image Here</p>
                            <p class="text-xs text-slate-500">Supports JPG, JPEG, PNG (Brain MRI Scans)</p>
                        </div>

                        <div id="preview-container" class="hidden space-y-3">
                            <img id="image-preview" class="max-h-64 mx-auto rounded-lg shadow-md border border-slate-700 object-cover" />
                            <p id="file-name" class="text-xs text-indigo-300 font-mono"></p>
                        </div>
                    </div>

                    <button id="analyze-btn" onclick="analyzeScan()" disabled 
                            class="w-full mt-6 py-3 px-6 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-white font-semibold rounded-xl shadow-lg transition duration-200 flex items-center justify-center space-x-2">
                        <i class="fa-solid fa-microscope"></i>
                        <span>Run AI Diagnosis</span>
                    </button>
                </div>

                <!-- Right Panel: Results -->
                <div class="bg-slate-800/50 border border-slate-700/60 rounded-2xl p-6 backdrop-blur flex flex-col justify-between">
                    <div>
                        <h2 class="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
                            <i class="fa-solid fa-square-poll-vertical text-emerald-400"></i> Diagnostic Analysis
                        </h2>

                        <!-- Placeholder before scan -->
                        <div id="results-placeholder" class="py-16 text-center space-y-3">
                            <i class="fa-solid fa-laptop-medical text-4xl text-slate-600"></i>
                            <p class="text-slate-400 text-sm">Upload an MRI image and click "Run AI Diagnosis" to view results.</p>
                        </div>

                        <!-- Results Content -->
                        <div id="results-content" class="hidden space-y-6">
                            <!-- Diagnosis Badge -->
                            <div class="bg-slate-900/80 border border-slate-700 rounded-xl p-5 text-center">
                                <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Primary Classification</span>
                                <div id="diagnosis-text" class="text-3xl font-extrabold text-indigo-400 mt-1 uppercase tracking-wide">--</div>
                                <div class="mt-2 text-xs text-slate-400">
                                    Confidence: <span id="confidence-text" class="font-bold text-emerald-400">0%</span>
                                </div>
                            </div>

                            <!-- Class Probabilities Breakdown -->
                            <div>
                                <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Probability Distribution</h3>
                                <div id="probability-bars" class="space-y-3"></div>
                            </div>
                        </div>
                    </div>

                    <div class="mt-6 pt-4 border-t border-slate-700/50 text-center">
                        <p class="text-[11px] text-slate-500">
                            <i class="fa-solid fa-circle-info mr-1"></i> For Screening Assistance Only. Always verify with a certified Neuro-Radiologist.
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let selectedFile = null;

            function handleFileSelect(event) {
                const file = event.target.files[0];
                if (file) {
                    selectedFile = file;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('image-preview').src = e.target.result;
                        document.getElementById('upload-prompt').classList.add('hidden');
                        document.getElementById('preview-container').classList.remove('hidden');
                        document.getElementById('file-name').innerText = file.name;
                        document.getElementById('analyze-btn').disabled = false;
                    }
                    reader.readAsDataURL(file);
                }
            }

            async function analyzeScan() {
                if (!selectedFile) return;

                const btn = document.getElementById('analyze-btn');
                btn.disabled = true;
                btn.innerHTML = `<i class="fa-solid fa-spinner animate-spin"></i> <span>Analyzing Brain Scan...</span>`;

                const formData = new FormData();
                formData.append("file", selectedFile);

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();

                    if (response.ok) {
                        displayResults(data);
                    } else {
                        alert("Error: " + (data.detail || "Analysis failed"));
                    }
                } catch (err) {
                    alert("Server error: " + err.message);
                } finally {
                    btn.disabled = false;
                    btn.innerHTML = `<i class="fa-solid fa-microscope"></i> <span>Run AI Diagnosis</span>`;
                }
            }

            function displayResults(data) {
                document.getElementById('results-placeholder').classList.add('hidden');
                document.getElementById('results-content').classList.remove('hidden');

                document.getElementById('diagnosis-text').innerText = data.diagnosis;
                document.getElementById('confidence-text').innerText = (data.confidence_score * 100).toFixed(1) + "%";

                const barsContainer = document.getElementById('probability-bars');
                barsContainer.innerHTML = '';

                Object.entries(data.class_probabilities).forEach(([cls, prob]) => {
                    const pct = (prob * 100).toFixed(1);
                    const isTop = cls === data.diagnosis;

                    const row = document.createElement('div');
                    row.className = 'space-y-1';
                    row.innerHTML = `
                        <div class="flex justify-between text-xs font-medium">
                            <span class="${isTop ? 'text-indigo-300 font-bold' : 'text-slate-400'} uppercase">${cls}</span>
                            <span class="${isTop ? 'text-emerald-400 font-bold' : 'text-slate-400'}">${pct}%</span>
                        </div>
                        <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-700/50">
                            <div class="h-full rounded-full transition-all duration-500 ${isTop ? 'bg-indigo-500 shadow-lg shadow-indigo-500/50' : 'bg-slate-700'}" style="width: ${pct}%"></div>
                        </div>
                    `;
                    barsContainer.appendChild(row);
                });
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model is not loaded. Error: {model_load_error}"
        )

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Upload JPEG or PNG.")

    try:
        contents = await file.read()
        processed_img = preprocess_mri(contents)

        predictions = model.predict(processed_img)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        return {
            "filename": file.filename,
            "diagnosis": CLASSES[predicted_class_idx],
            "confidence_score": round(confidence, 4),
            "class_probabilities": {CLASSES[i]: round(float(predictions[0][i]), 4) for i in range(len(CLASSES))}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")