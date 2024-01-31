# import threading
# import cv2
# from ultralytics import YOLO


# def run_tracker_in_thread(rtsp_url, model):
#     cap = cv2.VideoCapture(rtsp_url)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         results = model.track(source=frame, persist=True)
#         res_plotted = results[0].plot()
#         cv2.imshow('p', res_plotted)
#         if cv2.waitKey(1) == ord('q'):
#             break
#     cap.release()


# # Load the models
# model1 = YOLO('yolov8n.pt')
# model2 = YOLO('yolov8n-seg.pt')

# # Define the RTSP URLs for the trackers
# rtsp_url1 = 'rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1902'
# rtsp_url2 = 'rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/3202'

# # Create the tracker threads
# tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(rtsp_url1, model1), daemon=True)
# tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(rtsp_url2, model2), daemon=True)

# # Start the tracker threads
# tracker_thread1.start()
# tracker_thread2.start()

# # Wait for the tracker threads to finish
# tracker_thread1.join()
# tracker_thread2.join()

# # Clean up and close windows
# cv2.destroyAllWindows()

import threading
import cv2
from ultralytics import YOLO

def run_tracker_in_thread(rtsp_url, model):
    video = cv2.VideoCapture(rtsp_url)
    while True:
        ret, frame = video.read()
        if ret:
            results = model.track(source=frame, persist=True)
            res_plotted = results[0].plot()
            cv2.imshow(rtsp_url, res_plotted)
        if cv2.waitKey(1) == ord('q'):
            break

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Define the RTSP URLs for the feeds
rtsp_url1 = 'rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1902'
rtsp_url2 = 'rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/3202'

# Create the tracker threads
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(rtsp_url1, model), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(rtsp_url2, model), daemon=True)

# Start the tracker threads
tracker_thread1.start()
tracker_thread2.start()

# Wait for a key press to exit
cv2.waitKey(0)

# Clean up and close windows
cv2.destroyAllWindows()
