import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture(0)  # Use 0 for webcam, or 'assets/video.mp4' for video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    output = results.render()[0]  # Draw boxes
    cv2.imshow("YOLOv5 Object Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
