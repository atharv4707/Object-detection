from ultralytics import YOLO
from PIL import Image

# Load YOLOv8s model
model = YOLO('yolov8s.pt')

# Load and predict
img = 'assets/sample.jpg'
results = model(img)

# Print results
results.print()
results.show()
