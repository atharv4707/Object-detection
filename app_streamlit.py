import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image

# Dummy detection function (replace with your object detection model)
def detect_objects(frame):
    # Example: just draw a rectangle
    h, w, _ = frame.shape
    cv2.rectangle(frame, (50, 50), (w-50, h-50), (0, 255, 0), 3)
    return frame

st.set_page_config(page_title="Object Detector", page_icon="🤖", layout="wide")

st.title("🤖 Real-Time Object Detection")
st.write("Turn on your webcam and start detecting objects live.")

# Sidebar options
st.sidebar.header("⚙️ Options")
live_mode = st.sidebar.checkbox("Enable Live Webcam", value=True)

if live_mode:
    st.info("🎥 Webcam is ON. Please allow browser access.")

    # Streamlit camera input
    picture = st.camera_input("Capture from webcam (auto-refresh for live feed)", key="webcam")

    if picture:
        # Convert image to OpenCV format
        img = Image.open(picture)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run detection
        detected = detect_objects(frame)

        # Convert back to RGB for display
        detected_rgb = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
        st.image(detected_rgb, channels="RGB", caption="Live Detection")
else:
    st.warning("🚫 Webcam is OFF. Enable it from sidebar.")
