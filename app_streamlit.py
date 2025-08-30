import cv2
import streamlit as st
import tempfile
import os
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

st.title("🎥 Live Object Detection with Webcam")

# Checkbox to start/stop webcam
start_cam = st.checkbox("✅ Start Webcam")

# Capture photo option
capture_photo = st.button("📸 Capture Photo")

FRAME_WINDOW = st.image([])

if start_cam:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Failed to open webcam")
    else:
        st.success("✅ Webcam started. Uncheck to stop.")

        while start_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to grab frame from webcam.")
                break

            # Run YOLO on frame
            results = model(frame, stream=True)

            # Draw results
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{model.names[cls]} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Convert frame to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

            # Capture photo option
            if capture_photo:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                cv2.imwrite(temp_file.name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                st.success(f"📸 Photo captured and saved at {temp_file.name}")
                st.image(frame, caption="Captured Photo", use_column_width=True)
                capture_photo = False  # Reset button after one click

        cap.release()
        st.warning("🛑 Webcam stopped.")

else:
    st.info("☑️ Check the box above to start webcam.")
