from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.utils.coco import COCO_CLASSES
from ultralytics import YOLO
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
from pydantic import BaseModel
import io
import cv2
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="src/static"), name="static")

RTSP_URL = "rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/" + "1902"
cap = cv2.VideoCapture(RTSP_URL)
model = YOLO("./models/yolov8n.onnx")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(Path(BASE_DIR, "templates")))


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/get-stream")
async def get_stream(websocket: WebSocket, channel_id: str):
    await websocket.accept()
    try:
        while True:
            success, frame = cap.read()
            # frame = cv2.resize(frame, (640, 480))

            if not success:
                break
            else:
                _, buffer = cv2.imencode(".jpg", frame)
                await websocket.send_text("some text")
                await websocket.send_bytes(buffer.tobytes())
    except WebSocketDisconnect:
        print("Client disconnected")


class Request(BaseModel):
    channel_id: str


async def stream(channel_id: str):
    RTSP_URL = (
        "rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/" + channel_id
    )
    stream = VideoGear(source=RTSP_URL).start()
    server = NetGear()
    try:
        while True:
            frame = stream.read()

            # check for frame if Nonetype
            if frame is None:
                break

            # {do something with the frame here}

            # send frame to server
            server.send(frame)

    except:
        print("Client disconnected")
        stream.stop()
        server.close()
        return {"channel_id": channel_id, "status": "disconnected"}


async def generate_stream(stream_id: str):
    RTSP_URL = (
        "rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/" + stream_id
    )
    cap = cv2.VideoCapture(RTSP_URL)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            _, buffer = cv2.imencode(".jpg", frame)
            io_bytes = io.BytesIO(buffer)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + io_bytes.getvalue() + b"\r\n"
            )  # io_bytes.read()

    except Exception as e:
        print(e)
        cap.release()


@app.get("/video-feed/{stream_id}")
async def display_stream(stream_id: str):
    return StreamingResponse(
        # generate_stream(stream_id),
        stream(stream_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


"""
ReactPy Code Snippets

"""


from reactpy import component, html, run
from src.frontend.components.sidebar import Sidebar
from src.frontend.components.dashboard import Dashboard
from reactpy.backend.fastapi import configure

# from components.printbutton import PrintButton

css_url = "https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css"
bulma_css = html.link({"rel": "stylesheet", "href": css_url})


@component
def Layout():
    return html.div(
        {"class": "section"},
        [
            html.div(
                {"class": "container"},
                [
                    html.div(
                        {"class": "columns "},
                        [
                            html.div(
                                {"class": "column is-narrow"},
                                [
                                    Sidebar(),
                                ],
                            ),
                            html.div(
                                {"class": "column"},
                                [
                                    Dashboard(),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )


@component
def App():
    return html.div(bulma_css, Layout())


# configure(app, App)
