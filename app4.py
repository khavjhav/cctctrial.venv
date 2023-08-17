import numpy as np
import cv2
import time
import onnxruntime
from sort import *

# Load the ONNX model
onnx_model_path = "yolo-weights\yolov8m.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# Initialize video capture
rtsp_url = "rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1902"
cap = cv2.VideoCapture(rtsp_url)
cap.set(3, 1280)
cap.set(4, 720)

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.5)
limitsUp = [(264, 185), (337, 188), (434, 356), (301, 354)]
limitsDown = [(168, 201), (215, 193), (213, 359), (159, 357)]

totalCountUp = []
totalCountDown = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse coordinates:", x, y)

# Set up the windows
cv2.namedWindow("Image")
# Set the mouse callback function for the window
cv2.setMouseCallback("Image", mouse_callback)

frame_count = 0
start_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Preprocess the image for input to the ONNX model
    input_name = ort_session.get_inputs()[0].name
    # img_resized = cv2.resize(img, (640, 640))  # Resize the image to match model input size
    img_resized = cv2.resize(img, (640, 480))

    img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension

    # Perform inference using ONNX Runtime session
    outputs = ort_session.run(None, {input_name: img_input})

# Get the names of the output nodes
    output_names = [output.name for output in ort_session.get_outputs()]

# Print the output names and their shapes
    for i, name in enumerate(output_names):
     shape = ort_session.get_outputs()[i].shape
     print(f"Output {i}: Name={name}, Shape={shape}")

    # # Process the outputs (you need to adapt this based on your model's output format)
    # # For example, if the output is a tensor containing bounding box coordinates and scores:
    # boxes = outputs[0]  # Extract bounding box coordinates
    # scores = outputs[1]  # Extract confidence scores
    # class_ids = outputs[2]  # Extract class IDs
    
    # # Update tracker with detections
    # detections = []
    # for i, score in enumerate(scores[0]):
    #     if score > 0.5:  # Set your desired confidence threshold here
    #         x1, y1, x2, y2 = boxes[0][i]
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #         detections.append([x1, y1, x2, y2, score])

    # resultsTracker = tracker.update(np.array(detections))
    
    img_with_boxes = img.copy()

    # for result in resultsTracker:
    #     x1, y1, x2, y2, id = result
    #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #     w, h = x2 - x1, y2 - y1
    #     cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(img_with_boxes, f'ID: {int(id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # # Calculate and display FPS
    # frame_count += 1
    # elapsed_time = time.time() - start_time
    # fps = frame_count / elapsed_time
    # cv2.putText(img_with_boxes, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Image", cv2.resize(img_with_boxes, (0, 0), fx=1, fy=1))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
