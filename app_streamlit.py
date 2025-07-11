import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time

# --- App Config ---
st.set_page_config(page_title="VisionLive: Real-Time Object Detection", page_icon="🤖", layout="centered")

# --- Sidebar ---
st.sidebar.title("VisionLive")
st.sidebar.markdown("""
**Instructions:**
- Click **Start Webcam** to begin real-time object detection.
- Allow browser access to your webcam.
- Detected objects will be highlighted live.

**Credits:**
- Powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Built with [Streamlit](https://streamlit.io)
""")

# --- Main Area ---
st.markdown("""
# 🤖 VisionLive: Real-Time Object Detection
Experience real-time object detection directly in your browser using YOLOv8 and your webcam.
""")

# --- Load Model ---
@st.cache_resource(show_spinner=True)
def load_model():
    return YOLO('yolov8s.pt')

model = load_model()

# --- Webcam UI ---
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    run = st.toggle('🎥 Start Webcam', value=False, key='webcam_toggle')
    FRAME_WINDOW = st.empty()
    DETECTED_WINDOW = st.empty()
    st.caption("Live webcam feed with detected objects will appear here.")

if run:
    cap = cv2.VideoCapture(0)
    st.info("Webcam started. Uncheck to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame from webcam.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        annotated_frame = results[0].plot()
        FRAME_WINDOW.image(annotated_frame, channels="RGB", use_container_width=True, caption="Live Detection")

        # --- Detected Object Names + Count ---
        names = results[0].names  # class index to name mapping
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            from collections import Counter
            counts = Counter(class_ids)
            detected_list = [f"- {names[c]} ({counts[c]})" for c in counts]
            detected_md = "\n".join(detected_list)
            DETECTED_WINDOW.markdown(f"**Detected:**\n{detected_md}")
        else:
            DETECTED_WINDOW.markdown("**Detected:** None")
        # Streamlit needs to yield control to allow UI interaction
        if not st.session_state.get('webcam_toggle', True):
            break
    cap.release()
    st.success("Webcam stopped.")
    FRAME_WINDOW.info('Webcam is stopped. Click "Start Webcam" to begin.')
else:
    FRAME_WINDOW.info('Webcam is stopped. Click "Start Webcam" to begin.')

# --- Footer ---
st.markdown("""
---
<center><sub>Made with ❤️ by VisionLive Team | 2024</sub></center>
""", unsafe_allow_html=True)



