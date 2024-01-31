import cv2
import numpy as np
from deepsparse import compile_model
import time

# Initialize DeepSparse model
sparse_model = compile_model("models/yolov8n_half.onnx")

# Replace with your actual RTSP URL
RTSP_URL = "rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1902"

# Initialize video capture
cap = cv2.VideoCapture(RTSP_URL)

prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Prepare the input data
    input_data = np.expand_dims(frame, axis=0).astype(np.float32)

    # Run inference using DeepSparse
    output = sparse_model.run([input_data])

    # TODO: Process the output to draw bounding boxes, etc.
    # This part is specific to how the DeepSparse model's output is structured.

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))

    # Display FPS on the frame
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from deepsparse import compile_model
# import time

# # Initialize DeepSparse model
# sparse_model = compile_model("models/yolov8n_half.onnx")

# # Replace with your actual RTSP URL
# RTSP_URL = "rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1902"

# # Initialize video capture
# cap = cv2.VideoCapture(RTSP_URL)

# prev_frame_time = 0
# new_frame_time = 0
# font = cv2.FONT_HERSHEY_SIMPLEX

# # Set the expected dimensions based on the information you provided
# expected_height = 640
# expected_width = 640
# expected_channels = 3  # Assuming RGB images

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     # Resize the input frame to match the expected dimensions
#     resized_frame = cv2.resize(frame, (expected_width, expected_height))

#     # Transpose the dimensions to match [batch_size, channels, height, width] format
#     input_data = np.ascontiguousarray(resized_frame.transpose(2, 0, 1), dtype=np.float32)
#     input_data = np.expand_dims(input_data, axis=0)

#     # Run inference using DeepSparse
#     output = sparse_model.run([input_data])
#       # For example, assuming output contains bounding box information as output_bboxes
    
#      # Process the output for detections (assuming output contains class confidence scores)
#     confidence_threshold = 0.5
#     filtered_detections = []

#     for class_scores in output[0]:
#         for row_scores in class_scores:
#             for confidence in row_scores:
#                 if confidence > confidence_threshold:
#                     filtered_detections.append(confidence)

#     # Draw bounding boxes on the frame for filtered detections
#     for det_confidence in filtered_detections:
#         # Calculate bounding box coordinates and draw them
#         # Add the necessary code here to draw bounding boxes on the frame

#     # Calculate FPS and display it on the frame
#         new_frame_time = time.time()
#         fps = 1 / (new_frame_time - prev_frame_time)
#         prev_frame_time = new_frame_time
#         fps_text = f"FPS: {int(fps)}"
#         cv2.putText(frame, fps_text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

#     # Show the frame
#     cv2.imshow("frame", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release the video capture object
# cap.release()
# cv2.destroyAllWindows()






