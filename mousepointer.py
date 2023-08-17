import numpy as np
import cv2

# Set up the RTSP feed
rtsp = "rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1902"
cap = cv2.VideoCapture(rtsp)
cap.set(3, 1280)
cap.set(4, 720)

# Create a window to display the frame
cv2.namedWindow("Image")

# Define empty lists for new coordinates
newLimitsUp = []
newLimitsDown = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse coordinates:", x, y)
        if len(newLimitsUp) < 4:
            newLimitsUp.append((x, y))
        elif len(newLimitsDown) < 4:
            newLimitsDown.append((x, y))
        if len(newLimitsUp) == 4 and len(newLimitsDown) == 4:
            cv2.setMouseCallback("Image", None)  # Remove the mouse callback

# Set the mouse callback function for the window
cv2.setMouseCallback("Image", mouse_callback)

while True:
    success, img = cap.read()

    # Display the frame
    cv2.imshow("Image", img)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

# Print the coordinates for limitsUp and limitsDown
print("limitsUp:", newLimitsUp)
print("limitsDown:", newLimitsDown)
