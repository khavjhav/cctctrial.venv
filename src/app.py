from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from src.utils.coco import COCO_CLASSES
from ultralytics import YOLO
from pathlib import Path
import supervision as sv
import time
import cv2
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()
from src.routes import streamer


app = FastAPI()


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(Path(BASE_DIR, "templates")))


@app.get("/")
async def base(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})


app.include_router(streamer.router)
