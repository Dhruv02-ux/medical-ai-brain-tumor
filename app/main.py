"""Routing only — all logic lives in core/ and genai/."""
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import cv2, base64

from app.core.preprocessing import validate_and_load_image, preprocess_mri
from app.core.predict import predict_tumor
from app.core.model_loader import get_model
from app.core.gradcam import generate_heatmap, overlay_heatmap_on_image
from app.core.schemas import PredictionResponse
from app.genai.report_chain import generate_report
from app.genai.simplifier_chain import simplify_report
from app.genai.qa_chain import answer_question

app = FastAPI(title="NeuroScan AI")
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
    """Fail fast at boot if the model can't load, instead of on the first user request."""
    get_model()

@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = validate_and_load_image(file.content_type, contents)
        result = await run_in_threadpool(predict_tumor, image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"filename": file.filename, **result}

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    contents = await file.read()
    image = validate_and_load_image(file.content_type, contents)

    def _compute():
        processed = preprocess_mri(image)
        heatmap = generate_heatmap(get_model(), processed)
        return overlay_heatmap_on_image(heatmap, image)

    overlay = await run_in_threadpool(_compute)
    _, buffer = cv2.imencode(".png", overlay)
    return {"heatmap_base64": base64.b64encode(buffer).decode()}

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