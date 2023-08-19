from reactpy import component, html
from vidgear.gears import NetGear
import cv2
import io



def streamer_reciever():
    # define Netgear Client with `receive_mode = True` and default parameter
    client = NetGear(receive_mode=True)
    # loop over
    while True:
        # receive frames from network
        frame = client.recv()

        # check for received frame if Nonetype
        if frame is None:
            break

        _, buffer = cv2.imencode(".jpg", frame)
        io_bytes = io.BytesIO(buffer)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + io_bytes.getvalue() + b"\r\n"
        )


@component
def Dashboard():
    return html.div(
        {
            "class": "box",
        },
        [
            html.p({"class": "title is-5"}, "Dashboard"),
            html.hr(),
            html.video(
                {
                    "src": "https://youtu.be/Sx4xVyXHl60?feature=shared",
                    "controls": "controls",
                }
            ),
        ],
    )
