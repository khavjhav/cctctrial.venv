import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

rtsp ="rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1902"
# cap = cv2.VideoCapture("../Videos/kds.mp4")  # For Video
cap = cv2.VideoCapture(rtsp)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)
# cap = cv2.VideoCapture(0)  # For Webcam
model = YOLO("../Yolo-Weights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limitsUp = [(264, 185), (337, 188), (434, 356), (301, 354)]
limitsDown = [(168, 201), (215, 193), (213, 359), (159, 357)]

# limitsUp = [(602, 147), (602, 866), (711, 968), (716, 137)]
# limitsDown = [(0, 0), (0, 200), (640, 200), (640, 0)]

totalCountUp = []
totalCountDown = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse coordinates:", x, y)
# Set up the windows
cv2.namedWindow("Image")
# Set the mouse callback function for the window
cv2.setMouseCallback("Image", mouse_callback)
while True:
    success, img = cap.read()
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    overlay_width = imgGraphics.shape[1]
    overlay_height = imgGraphics.shape[0]
    new_overlay_width = int(overlay_width * 0.3)
    new_overlay_height = int(overlay_height * 0.3)
    resized_imgGraphics = cv2.resize(imgGraphics, (new_overlay_width, new_overlay_height))

    img = cvzone.overlayPNG(img, resized_imgGraphics, (0, 0))
    results = model(img, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "person" and conf > 0.2:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    resultsTracker = tracker.update(detections)
    cv2.polylines(img, [np.array(limitsUp)], True, (0, 0, 255), 5)
    cv2.polylines(img, [np.array(limitsDown)], True, (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if cv2.pointPolygonTest(np.array(limitsUp), (cx, cy), False) == 1:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.polylines(img, [np.array(limitsUp)], True, (0, 255, 0), 5)
        if cv2.pointPolygonTest(np.array(limitsDown), (cx, cy), False) == 1:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.polylines(img, [np.array(limitsDown)], True, (0, 255, 0), 5)
    # cv2.putText(img, str(len(totalCountUp)), (60, 23), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    # cv2.putText(img, str(len(totalCountDown)), (148, 20), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)
    cv2.putText(img, str(len(totalCountUp)), (135, 26), cv2.FONT_HERSHEY_PLAIN, 2, (139, 195, 75), 3)
    cv2.putText(img, str(len(totalCountDown)), (60, 26), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 230), 3)


    cv2.imshow("Image", cv2.resize(img, (0, 0), fx=1, fy=1))
    cv2.waitKey(1)