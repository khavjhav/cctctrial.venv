from flask import Flask, render_template, Response
import cv2
import imutils
from ultralytics  import YOLO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    # Replace this with your RTSP feed URL
    # rtsp_url = 'https://10.221.219.84:8080'
    rtsp_url = 'rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1901'
    model=YOLO("yolov8n.pt")


    cap = cv2.VideoCapture(rtsp_url)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame=cv2.resize(frame,(640,480))
        cv2.imshow('Video Feed', frame)
    
        model.predict(frame,save=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    # cap = cv2.VideoCapture(rtsp_url)

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     # Perform people counting logic on the frame
    #     # You need to implement this part using computer vision techniques

    #     ret, buffer = cv2.imencode('.jpg', frame)
    #     if not ret:
    #         break

    #     frame = buffer.tobytes()
    #     yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
