import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter

# --- App Config ---
st.set_page_config(page_title="VisionLive: Real-Time Object Detection", page_icon="🤖", layout="wide")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        .main {background-color: #0D1117;}
        h1 {
            background: linear-gradient(to right, #00FFA3, #DC1FFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-size: 2.5rem;
        }
        .card {
            padding: 15px;
            border-radius: 10px;
            background-color: #161B22;
            color: white;
            margin-bottom: 15px;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=180)
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    st.markdown("---")
    st.markdown("👨‍💻 Powered by [YOLOv8](https://github.com/ultralytics/ultralytics)")
    st.markdown("🖥️ Built with [Streamlit](https://streamlit.io)")

# --- Title ---
st.markdown("<h1>🤖 VisionLive: Real-Time Object Detection</h1>", unsafe_allow_html=True)
st.caption("Experience real-time object detection in your browser using YOLOv8 and your webcam.")

# --- Instructions Card ---
st.markdown("""
<div class="card">
    <h3>📌 Instructions</h3>
    <ul>
        <li>Click <b>Start Webcam</b> to begin real-time detection.</li>
        <li>Allow browser access to your webcam.</li>
        <li>Detected objects will appear with labels below.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource(show_spinner=True)
def load_model():
    return YOLO('yolov8s.pt')

model = load_model()

# --- Webcam UI ---
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run = st.toggle('🎥 Start Webcam', value=False, key='webcam_toggle')
    FRAME_WINDOW = st.empty()
    DETECTED_WINDOW = st.empty()   # <- single container for results

# --- Live Detection ---
if run:
    cap = cv2.VideoCapture(0)
    st.info("✅ Webcam started. Uncheck to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to grab frame from webcam.")
            break

        # Run YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, conf=conf_threshold)
        annotated_frame = results[0].plot()
        FRAME_WINDOW.image(annotated_frame, channels="RGB", use_container_width=True, caption="📹 Live Detection")

        # --- Detected Object Counts ---
        names = results[0].names
        boxes = results[0].boxes

        with DETECTED_WINDOW.container():
            if boxes is not None and len(boxes) > 0:
                class_ids = boxes.cls.cpu().numpy().astype(int)
                counts = Counter(class_ids)

                # Show metrics in 3 columns
                cols = st.columns(3)
                for i, (cls, count) in enumerate(counts.items()):
                    label = names[cls].capitalize()
                    cols[i % 3].metric(label, count)

                # Single clean summary line
                detected_summary = " | ".join([f"{names[c].capitalize()}: {counts[c]}" for c in counts])
                st.markdown(f"### 🔎 Detected: {detected_summary}")
            else:
                st.markdown("### 🔎 Detected: None ❌")

        # Break if toggle off
        if not st.session_state.get('webcam_toggle', True):
            break

    cap.release()
    st.success("🛑 Webcam stopped.")
else:
    FRAME_WINDOW.info('Webcam is stopped. Click "Start Webcam" to begin.')

# --- Footer ---
st.markdown("""
---
<center><sub>🚀 Made with ❤️ by Atharv | VisionLive 2025</sub></center>
""", unsafe_allow_html=True)
