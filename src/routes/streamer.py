from fastapi import APIRouter
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import os
from pathlib import Path
from vidgear.gears import NetGear
from dotenv import load_dotenv

load_dotenv()

HOST_IP = os.getenv("HOST_IP")

router = APIRouter(
    prefix="/api/v1",
)
options = {"multiclient_mode": True}
camera_port_mappings = {
    "1902": "5567",
}


def gen_frames(channel_id: str):
    rtps_url = f"{os.getenv('RTSP_ROOT_URL')}{channel_id}"
    print(rtps_url)
    camera = cv2.VideoCapture(rtps_url)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# async def gen_frames(channel_id: str):
#     client = NetGear(
#         address=str(HOST_IP),
#         port=camera_port_mappings[channel_id],
#         protocol="tcp",
#         pattern=2,
#         receive_mode=True,
#         logging=True,
#         **options,
#     )
#     # loop over
#     while True:
#         # receive data from server
#         frame = client.recv()

#         # check for frame if None
#         if frame is None:
#             client.close()
#             break

#         _, buffer = cv2.imencode(".jpg", frame)
#         frame = buffer.tobytes()
#         yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@router.get("/stream/{channel_id}")
async def get_stream(channel_id: str):
    return StreamingResponse(
        gen_frames(channel_id), media_type="multipart/x-mixed-replace;boundary=frame"
    )


@router.websocket("/ws-get-stream-1/{channel_id}")
async def ws_get_stream_1(websocket: WebSocket, channel_id: str):
    rtps_url = f"{os.getenv('RTSP_ROOT_URL')}{channel_id}"
    await websocket.accept()
    try:
        camera = cv2.VideoCapture(rtps_url)
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                break
            else:
                _, buffer = cv2.imencode(".jpg", frame)
                await websocket.send_bytes(buffer.tobytes())
    except WebSocketDisconnect:
        print("Client disconnected")
