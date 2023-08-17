# from flask import Flask, render_template, Response
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import math
# import imutils

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')


# def generate():
    
#     rtsp_url = 'rtsp://192.168.137.242:8080/h264_pcm.sdp'
#     cap = cv2.VideoCapture(rtsp_url)  # Use the laptop webcam
#     cap.set(3, 640)  # Set width to 640
#     cap.set(4, 480)  # Set height to 480

#     model = YOLO("../Yolo-Weights/yolov8n-seg.pt")

#     while cap.isOpened():
#         success, img = cap.read()
#         if not success:
#             break

#         # Object detection and counting logic
#         results = model(img)

#         # Process results and draw bounding boxes
#         for r in results.pred[0]:
#             class_idx = r.argmax()
#             x1, y1, x2, y2 = r[:4].int().tolist()
#             img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             img = cv2.putText(img, f'{model.names[class_idx]}', (x1, y1 - 10),
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         ret, buffer = cv2.imencode('.jpg', img)
#         if not ret:
#             break

#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)



# v2

from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    rtsp_url = 'rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1901'
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(3, 640)
    cap.set(4, 480)

    model = YOLO("../yolo-weights/yolov8n.pt")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        results = model(img)

        if len(results.pred[0]) > 0:
            detections = results.pred[0].numpy()

            for r in detections:
                class_idx = int(r[5])
                label = model.names[class_idx]
                x1, y1, x2, y2 = map(int, r[:4])

                if label == "person":
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    img = cv2.putText(img, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print(f"Detected: {label}, Bounding Box: ({x1}, {y1}) - ({x2}, {y2})")

        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            break

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

