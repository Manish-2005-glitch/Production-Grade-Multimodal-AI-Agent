from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from prometheus_client import generate_latest
from fastapi.responses import Response
from agent import agent_run
from Metrics import REQUEST_COUNT, REQUEST_LATENCY

app = FastAPI(
    title = "VisionRAG-Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/agent")
async def run_agent(
    question:str = Form(...),
    file: UploadFile = File(None)
):
    REQUEST_COUNT.inc()
    
    image = None
    
    if file:
        img_bytes = await file.read()
        image = cv2.imdecode(
            np.frombuffer(img_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
        
    with REQUEST_LATENCY.time():
        response = agent_run(question, image)
        
    return {
        "question" : question,
        "response" : response
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")