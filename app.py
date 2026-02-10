from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import numpy as np
import cv2
import base64
from video_tracker import process_video
import tempfile
from fastapi.responses import FileResponse

from prometheus_client import generate_latest
from Metrics import REQUEST_COUNT, REQUEST_LATENCY
from agent import agent_run


app = FastAPI(
    title="VisionRAG-Agent",
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
    question: str = Form(...),
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
        result = agent_run(question, image)

    response = {
        "question": question,
        "response": result["answer"]
    }

    if result.get("annotated_image") is not None:
        _, buffer = cv2.imencode(".jpg", result["annotated_image"])
        response["image"] = base64.b64encode(buffer).decode("utf-8")

    return response

@app.post("/video")
async def run_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    output_path = input_path.replace(".mp4", "_tracked.mp4")

    process_video(input_path, output_path)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="tracked_output.mp4"
    )
    
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
