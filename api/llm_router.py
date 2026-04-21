from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest
from PIL import Image
import uuid
import traceback
import os
import tempfile

from api.embedding_logger import log_query, log_multimodal_event
from pipeline.image_embedding import OpenCLIPImageEmbedder
from pipeline.video_embedding import embed_video_file
from monitoring.retrain_trigger import evaluate_retrain_need
from monitoring.dashboard_data import get_summary, get_charts
from scheduler.retrain_job import (
    start_retrain_scheduler,
    stop_retrain_scheduler,
    scheduled_retrain_check,
)
from metrics.monitoring_metrics import (
    query_counter,
    cheap_model_usage,
    mid_model_usage,
    expert_model_usage,
)

app = FastAPI(
    title="Drift-Aware LLMOps API",
    version="1.0.0"
)

# CORS (Next.js frontend on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded embedder
image_embedder = None


def get_image_embedder():
    global image_embedder
    if image_embedder is None:
        image_embedder = OpenCLIPImageEmbedder(
            model_name="RN50",
            pretrained="openai"
        )
    return image_embedder


# -----------------------------
# App lifecycle
# -----------------------------
@app.on_event("startup")
def on_startup():
    start_retrain_scheduler(hours=6)


@app.on_event("shutdown")
def on_shutdown():
    stop_retrain_scheduler()


# -----------------------------
# Health / Root
# -----------------------------
@app.get("/", operation_id="root_get")
def root():
    return {"status": "ok", "service": "drift-aware-llmops"}


@app.get("/health", operation_id="health_get")
def health():
    return {"status": "healthy"}


# -----------------------------
# Simulated LLM Models
# -----------------------------
def cheap_model(query: str):
    return f"[Cheap Model] Basic answer for: {query}"


def mid_model(query: str):
    return f"[Mid Model] Detailed explanation for: {query}"


def expert_model(query: str):
    return f"[Expert Model] Advanced reasoning for: {query}"


# -----------------------------
# Difficulty Estimator
# -----------------------------
def estimate_difficulty(query: str):
    length = len(query)
    if length < 40:
        return "easy"
    elif length < 120:
        return "medium"
    else:
        return "hard"


# -----------------------------
# LLM Router Endpoint (text)
# -----------------------------
@app.post("/query", operation_id="query_post")
def route_query(query: str):
    query_counter.inc()
    difficulty = estimate_difficulty(query)

    if difficulty == "easy":
        response = cheap_model(query)
        model_used = "cheap_model"
        cheap_model_usage.inc()
    elif difficulty == "medium":
        response = mid_model(query)
        model_used = "mid_model"
        mid_model_usage.inc()
    else:
        response = expert_model(query)
        model_used = "expert_model"
        expert_model_usage.inc()

    # Existing text logger
    log_query(query)

    return {
        "model": model_used,
        "difficulty": difficulty,
        "response": response
    }


# -----------------------------
# Image Embedding Endpoint
# -----------------------------
@app.post("/embed-image", operation_id="embed_image_post")
async def embed_image(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())

    try:
        image = Image.open(file.file).convert("RGB")
        embedder = get_image_embedder()
        embedding = embedder.embed_pil_image(image)

        log_multimodal_event(
            request_id=request_id,
            modality="image",
            input_ref=file.filename,
            embedding=embedding,
            model_name=f"image_embedder:{embedder.mode}",
            model_response="image embedding stored"
        )

        return {
            "request_id": request_id,
            "status": "success",
            "modality": "image",
            "embedder_mode": embedder.mode,
            "embedding_dim": len(embedding)
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image embedding failed: {str(e)}")


# -----------------------------
# Video Embedding Endpoint
# -----------------------------
@app.post("/embed-video", operation_id="embed_video_post")
async def embed_video(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        embedder = get_image_embedder()
        embedding = embed_video_file(
            video_path=tmp_path,
            image_embedder=embedder,
            sample_fps=1.0,
            max_frames=24
        )

        log_multimodal_event(
            request_id=request_id,
            modality="video",
            input_ref=file.filename,
            embedding=embedding,
            model_name=f"video_embedder:{embedder.mode}",
            model_response="video embedding stored"
        )

        return {
            "request_id": request_id,
            "status": "success",
            "modality": "video",
            "embedder_mode": embedder.mode,
            "embedding_dim": len(embedding)
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video embedding failed: {str(e)}")

    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# -----------------------------
# Retrain Endpoints
# -----------------------------
@app.get("/retrain/evaluate", operation_id="retrain_evaluate_get")
def retrain_evaluate():
    return evaluate_retrain_need()


@app.post("/retrain/run-now", operation_id="retrain_run_now_post")
def retrain_run_now():
    scheduled_retrain_check()
    return {"status": "ok", "message": "Manual retrain check executed"}


# -----------------------------
# Dashboard Endpoints
# -----------------------------
@app.get("/dashboard/summary", operation_id="dashboard_summary_get")
def dashboard_summary():
    return get_summary()


@app.get("/dashboard/charts", operation_id="dashboard_charts_get")
def dashboard_charts():
    return get_charts()


# -----------------------------
# Prometheus Metrics Endpoint
# -----------------------------
@app.get("/metrics", operation_id="metrics_get")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {"status": "ok", "service": "drift-aware-llmops"}
