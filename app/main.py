"""Routing only — all logic lives in core/ and genai/."""
import os
import cv2
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.core.preprocessing import validate_and_load_image, preprocess_mri
from app.core.predict import predict_tumor
from app.core.model_loader import get_model
from app.core.gradcam import generate_heatmap, overlay_heatmap_on_image, render_heatmap_colormap, get_grad_model
from app.core.schemas import PredictionResponse
from app.genai.report_chain import generate_report
from app.genai.simplifier_chain import simplify_report
from app.genai.qa_chain import answer_question

app = FastAPI(title="NeuroScan AI", description="Clinical Decision Support & Explainable Brain MRI Diagnostics")
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class ReportRequest(BaseModel):
    diagnosis: str
    confidence_score: float
    class_probabilities: dict[str, float]

class SimplifyRequest(BaseModel):
    report: str

class QuestionRequest(BaseModel):
    question: str

@app.on_event("startup")
async def preload_model() -> None:
    """Pre-warm model graph and Grad-CAM sub-model for sub-50ms user latency."""
    try:
        model = get_model()
        get_grad_model(model)
    except Exception as e:
        print(f"[Startup Warning] Model warm-up error: {e}")

@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    include_heatmap: bool = Query(True, description="Pre-generate Grad-CAM heatmap in response")
):
    contents = await file.read()
    try:
        image = validate_and_load_image(file.content_type, contents)
        result = await run_in_threadpool(predict_tumor, image)
        
        heatmap_base64 = None
        pure_heatmap_base64 = None
        
        if include_heatmap:
            def _gen_cam():
                model = get_model()
                processed = preprocess_mri(image)
                hm = generate_heatmap(model, processed)
                overlay = overlay_heatmap_on_image(hm, image, alpha=0.55, colormap_name="turbo")
                pure = render_heatmap_colormap(hm, colormap_name="turbo")
                
                # Convert RGB to BGR for cv2.imencode to preserve exact color channels
                _, b_overlay = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                _, b_pure = cv2.imencode(".png", cv2.cvtColor(pure, cv2.COLOR_RGB2BGR))
                return base64.b64encode(b_overlay).decode(), base64.b64encode(b_pure).decode()

            heatmap_base64, pure_heatmap_base64 = await run_in_threadpool(_gen_cam)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "filename": file.filename or "scan.jpg",
        "diagnosis": result["diagnosis"],
        "confidence_score": result["confidence_score"],
        "class_probabilities": result["class_probabilities"],
        "low_confidence_flag": result["low_confidence_flag"],
        "tumor_info": result.get("tumor_info"),
        "differential": result.get("differential"),
        "heatmap_base64": heatmap_base64,
        "pure_heatmap_base64": pure_heatmap_base64,
    }

@app.post("/gradcam")
async def gradcam(
    file: UploadFile = File(...),
    colormap: str = Query("turbo", description="Colormap: turbo, jet, inferno, viridis"),
    alpha: float = Query(0.55, description="Overlay opacity 0.0 to 1.0"),
    threshold: float = Query(0.10, description="Minimum activation threshold")
):
    contents = await file.read()
    try:
        image = validate_and_load_image(file.content_type, contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def _compute():
        model = get_model()
        processed = preprocess_mri(image)
        heatmap = generate_heatmap(model, processed)
        overlay = overlay_heatmap_on_image(heatmap, image, alpha=alpha, colormap_name=colormap, threshold=threshold)
        pure = render_heatmap_colormap(heatmap, colormap_name=colormap)
        
        _, b_overlay = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        _, b_pure = cv2.imencode(".png", cv2.cvtColor(pure, cv2.COLOR_RGB2BGR))
        return base64.b64encode(b_overlay).decode(), base64.b64encode(b_pure).decode()

    overlay_b64, pure_b64 = await run_in_threadpool(_compute)
    return {
        "heatmap_base64": overlay_b64,
        "pure_heatmap_base64": pure_b64,
        "colormap": colormap,
        "alpha": alpha,
    }

@app.get("/samples/{sample_name}")
async def get_sample_scan(sample_name: str):
    """Provides authentic clinical brain MRI test scans for 1-click live demo."""
    clean_name = os.path.basename(sample_name).replace(".jpg", "") + ".jpg"
    path = os.path.join("app/static/samples", clean_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample scan not found")
    return FileResponse(path, media_type="image/jpeg")

@app.post("/generate-report")
async def report_endpoint(req: ReportRequest):
    report = await run_in_threadpool(generate_report, req.diagnosis, req.confidence_score, req.class_probabilities)
    return {"report": report}

@app.post("/simplify-report")
async def simplify_endpoint(req: SimplifyRequest):
    return {"simplified_report": await run_in_threadpool(simplify_report, req.report)}

@app.post("/ask")
async def ask_endpoint(req: QuestionRequest):
    return {"answer": await run_in_threadpool(answer_question, req.question)}
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
