import streamlit as st
import cv2
import pandas as pd
import datetime
import numpy as np
import tempfile
import os
import easyocr
import time 
import torch
import threading
from ultralytics import YOLO
import alarm_utils  # Keep using the corrected alarm_utils.py from the previous step

import uuid


audio_path = os.path.join(os.path.dirname(__file__), "Alarm-2-chosic.com_.mp3")

import base64

def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()








# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RNO (AI) - Intelligent Vision Platform",
    page_icon="🤖",
    layout="wide"
)

# --- Splash Screen ---
if "splash_done" not in st.session_state:
    st.markdown(
        """
        <div style='text-align:center; padding-top:180px;'>
            <h1 style='font-size:3em; color:#00C4FF;'>🤖 Welcome to <b>RNO (AI)</b></h1>
            <p style='font-size:1.2em; color:#ccc;'>Initializing intelligent vision systems...</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    st.session_state.splash_done = True
    st.rerun()  # <--- updated for latest Streamlit versions


# --- Main Dashboard UI ---
st.title("RNO (AI) - Intelligent Vision Platform")
st.write("Welcome! The dashboard is ready. Models will load only when you use a module.")








# --- PAGE CONFIGURATION (Unchanged) ---
st.set_page_config(
    page_title="Intelligent Vision Platform",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODERN UI STYLES (Unchanged) ---
st.markdown("""
    <style>
        /* CSS styles remain the same */
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .main .block-container { padding: 2rem 5rem; }
        .header-title { font-size: 48px !important; font-weight: 700; color: #1E3A8A; text-align: center; margin-bottom: 10px; }
        .header-subtitle { font-size: 20px; color: #6B7280; text-align: center; margin-bottom: 2rem; }
        .st-emotion-cache-16txtl3 { padding: 2rem 1rem; }
        .st-emotion-cache-16txtl3 h1 { color: #1E3A8A; font-weight: 700; }
        .module-card { background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 25px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); transition: all 0.3s ease-in-out; }
        .module-card:hover { box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); border-left: 5px solid #2563EB; }
        .stButton>button { border-radius: 8px; border: 1px solid #2563EB; color: #2563EB; background-color: transparent; padding: 10px 20px; font-weight: bold; transition: all 0.3s; }
        .stButton>button:hover { background-color: #2563EB; color: white; border-color: #2563EB; }
        .footer { text-align: center; color: #9CA3AF; font-size: 14px; padding-top: 40px; }
    </style>
""", unsafe_allow_html=True)


# --- MODEL CACHING (Unchanged) ---
@st.cache_resource
def load_yolo_model(model_path):
    if not os.path.exists(model_path): st.error(f"❌ Model file not found: '{model_path}'."); return None
    try: return YOLO(model_path)
    except Exception as e: st.error(f"⚠️ Error loading model: {e}"); return None

@st.cache_resource
def load_yolov5_model_from_hub(model_path, class_names):
    if not os.path.exists(model_path): st.error(f"❌ Model file not found: '{model_path}'."); return None
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='github')
        model.names = class_names
        return model
    except Exception as e: st.error(f"⚠️ Error loading YOLOv5 model: {e}"); return None

@st.cache_resource
def load_anpr_models():
    with st.spinner("Initializing ANPR models..."):
        vehicle_model, lp_detector = load_yolo_model('yolov8n.pt'), load_yolo_model('license_plate_detector.pt')
        try: ocr_reader = easyocr.Reader(['en'])
        except Exception as e: st.error(f"⚠️ Error initializing EasyOCR: {e}"); return None, None, None
    return vehicle_model, lp_detector, ocr_reader

# --- DRAWING FUNCTIONS (Unchanged) ---
def draw_fire_boxes(img, boxes, classes, scores, names):
    for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]); label = f"{names.get(int(cls), str(int(cls)))}: {conf:.2f}"; color = (0, 0, 255); cv2.rectangle(img, (x1, y1), (x2, y2), color, 2); (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1); cv2.rectangle(img, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1); cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

def draw_ppe_boxes(img, boxes, classes, scores, names):
    for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]); class_id = int(cls); label = f"{names.get(class_id, str(class_id))}: {conf:.2f}"; class_name = names.get(class_id, "").lower(); color = (0, 0, 255) if "no-" in class_name else (0, 255, 0); cv2.rectangle(img, (x1, y1), (x2, y2), color, 2); (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1); cv2.rectangle(img, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1); cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img

def draw_gun_boxes(img, boxes, classes, scores, names):
    for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]); class_id = int(cls); label = f"{names[class_id]}: {conf:.2f}"; color = (255, 0, 0); cv2.rectangle(img, (x1, y1), (x2, y2), color, 2); (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2); cv2.rectangle(img, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1); cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img

# --- FRAME PROCESSING (Functions now trigger alerts) (Unchanged) ---
def process_frame_with_model(model, frame, conf, draw_function, alert_module_name=None):
    if model is None: return frame
    results = model.predict(frame, conf=conf, imgsz=640, verbose=False)
    r = results[0]
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        boxes, classes, scores = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()
        frame = draw_function(frame, boxes, classes, scores, model.names)
        if alert_module_name: alarm_utils.trigger_alert(alert_module_name)
    return frame

def process_frame_with_yolov5_model(model, frame, conf, draw_function, alert_module_name=None):
    if model is None: return frame
    model.conf = conf
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame, size=640)
    detections = results.xyxy[0]
    if len(detections) > 0:
        boxes, scores, classes = detections[:, :4].cpu().numpy(), detections[:, 4].cpu().numpy(), detections[:, 5].cpu().numpy()
        frame = draw_function(frame, boxes, classes, scores, model.names)
        if alert_module_name: alarm_utils.trigger_alert(alert_module_name)
    return frame

# --- ANPR and other functions (Unchanged) ---
def find_closest_car(lp_box, car_boxes):
    if not car_boxes: return None; lp_center_x = (lp_box[0] + lp_box[2]) / 2; lp_center_y = (lp_box[1] + lp_box[3]) / 2; min_dist = float('inf'); closest_car_box = None
    for car_box in car_boxes:
        car_center_x, car_center_y = (car_box[0] + car_box[2]) / 2, (car_box[1] + car_box[3]) / 2; dist = np.sqrt((lp_center_x - car_center_x)**2 + (lp_center_y - car_center_y)**2)
        if dist < min_dist: min_dist, closest_car_box = dist, car_box
    return closest_car_box

def process_frame_for_anpr(frame, vehicle_model, lp_detector, ocr_reader, conf):
    if not all([vehicle_model, lp_detector, ocr_reader]): st.warning("ANPR models not loaded."); return frame
    vehicle_detections = vehicle_model(frame, classes=[2, 3, 5, 7], verbose=False)[0]; car_boxes = [box.xyxy.cpu().numpy().flatten().tolist() for box in vehicle_detections.boxes]
    lp_detections = lp_detector(frame, conf=conf, verbose=False)[0]
    for lp_box_obj in lp_detections.boxes:
        lp_box = lp_box_obj.xyxy.cpu().numpy().flatten().tolist(); x1, y1, x2, y2 = map(int, lp_box)
        associated_car_box = find_closest_car(lp_box, car_boxes)
        if associated_car_box: car_x1, car_y1, car_x2, car_y2 = map(int, associated_car_box); cv2.rectangle(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 3)
        license_plate_crop = frame[y1:y2, x1:x2]
        if license_plate_crop.size > 0:
            lp_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY); _, lp_thresh = cv2.threshold(lp_gray, 64, 255, cv2.THRESH_BINARY_INV)
            ocr_results = ocr_reader.readtext(lp_thresh)
            if ocr_results: text = ocr_results[0][1]; cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3); cv2.putText(frame, f'{text.upper().replace(" ", "")}', (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
    return frame

# --- SIDEBAR (Unchanged) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004531.png", width=100)
    st.title("Intelligent Vision Platform")
    st.markdown("---")
    module = st.radio(
        "**Select a Monitoring Module**",
        ["🦺 Workshop Safety", "🔫 Gun Detection", "🔥 Smoke & Fire Detection", "🏎 Vehicle & Plate Recognition (ANPR)"],
        captions=["Monitor PPE compliance", "Detects firearms and other weapons", "Detect fire and smoke in real-time", "Automated Number Plate Recognition"]
    )
    st.markdown("---")
    st.info("Navigate through different AI-powered vision modules using the options above.")

# --- HEADER (Unchanged) ---
st.markdown('<p class="header-title">Intelligent Vision Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">An AI-Powered Dashboard for Real-Time Visual Monitoring</p>', unsafe_allow_html=True)



def process_video_or_webcam(module_name, process_function):
    # --- Initialize states ---
    if "show_alert" not in st.session_state:
        st.session_state.show_alert = False
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "input_type" not in st.session_state:
        st.session_state.input_type = None

    col1, col2 = st.columns(2)
    with col1:
        input_type = st.radio(
            "**Choose Input Source**",
            ["📁 Upload Video", "📷 Use Webcam"],
            key=f"input_{module_name}",
            horizontal=True
        )

    alert_placeholder = st.empty()
    frame_placeholder = st.empty()

    # Update selected input source
    st.session_state.input_type = input_type

    # ---------------- VIDEO UPLOAD ----------------
    if input_type == "📁 Upload Video":
        uploaded_video = st.file_uploader(
            "Upload a video file (.mp4, .avi, .mov)",
            type=["mp4", "avi", "mov"],
            key=f"upload_{module_name}"
        )

        if uploaded_video:
            if st.session_state.cap is None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                st.session_state.cap = cv2.VideoCapture(tfile.name)
                st.session_state.temp_file = tfile.name

            cap = st.session_state.cap
            st.info("Processing uploaded video...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.success("✅ Video processing complete.")
                    break

                processed_frame = process_function(frame)
                frame_placeholder.image(
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                    use_column_width=True
                )

                # Handle alert state
                if st.session_state.active_alert_module and not st.session_state.show_alert:
                    st.session_state.show_alert = True

                if st.session_state.show_alert:
                    alert_module = st.session_state.active_alert_module
                    with alert_placeholder.container():
                        st.error(
                            f"🚨 {alert_module.upper()} DETECTED! Review footage and take action.",
                            icon="🔥" if alert_module == "Fire" else "🔫"
                        )

                        if st.button("Dismiss Alert", key=f"dismiss_{module_name}_video_{uuid.uuid4()}"):
                            st.session_state.show_alert = False
                            st.session_state.active_alert_module = None
                            st.stop()  # Stop UI update only, not video loop

            cap.release()
            os.remove(st.session_state.temp_file)
            st.session_state.cap = None

    # ---------------- WEBCAM FEED ----------------
    elif input_type == "📷 Use Webcam":
        if st.button("▶️ Start Webcam Feed", key=f"start_webcam_{module_name}"):
            if st.session_state.cap is None:
                st.session_state.cap = cv2.VideoCapture(0)

            cap = st.session_state.cap
            if not cap.isOpened():
                st.error("❌ Could not open webcam.")
                return

            st.warning("Webcam active. Click Dismiss Alert to hide alerts.", icon="📷")

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Webcam feed ended.")
                    break

                processed_frame = process_function(frame)
                frame_placeholder.image(
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                    use_column_width=True
                )

                if st.session_state.active_alert_module and not st.session_state.show_alert:
                    st.session_state.show_alert = True

                if st.session_state.show_alert:
                    alert_module = st.session_state.active_alert_module
                    with alert_placeholder.container():
                        st.error(
                            f"🚨 {alert_module.upper()} DETECTED! Review footage and take action.",
                            icon="🔥" if alert_module == "Fire" else "🔫"
                        )
                        if st.button("Dismiss Alert", key=f"dismiss_{module_name}_webcam_{uuid.uuid4()}"):
                            st.session_state.show_alert = False
                            st.session_state.active_alert_module = None
                            st.stop()

            cap.release()
            st.session_state.cap = None




# --- MODULES DISPLAY (with default conf 0.6) ---
with st.container():
    st.markdown('<div class="module-card">', unsafe_allow_html=True)

    if module == "🦺 Workshop Safety":
        st.subheader("🦺 Workshop Safety Monitoring")
        st.info("Monitors for Personal Protective Equipment (PPE) compliance.", icon="ℹ️")
        PPE_MODEL_PATH = "ppe_best.pt"
        ppe_model = load_yolo_model(PPE_MODEL_PATH)
        if ppe_model:
            st.success(f"✅ PPE Model '{PPE_MODEL_PATH}' loaded.")
            with st.expander("View Detected Classes"): st.write(list(ppe_model.names.values()))
            conf = st.slider("Confidence", 0.1, 1.0, 0.6, 0.05, key="ppe_conf")  # <-- changed default to 0.6
            process_video_or_webcam("Workshop Safety", lambda frame: process_frame_with_model(ppe_model, frame, conf, draw_ppe_boxes))

    elif module == "🔫 Gun Detection":
        st.subheader("🔫 Gun & Weapon Detection")
        st.info("Detects weapons. An email and audio alert will be triggered instantly if a weapon is detected.", icon="🚨")
        GUN_MODEL_PATH = "yolov5s_my.pt"
        GUN_CLASS_NAMES = ['knife', 'pistol']
        gun_model = load_yolov5_model_from_hub(GUN_MODEL_PATH, GUN_CLASS_NAMES)
        if gun_model:
            st.success(f"✅ Weapon Model '{GUN_MODEL_PATH}' loaded.")
            with st.expander("View Detected Classes"): st.write(gun_model.names)
            conf = st.slider("Confidence", 0.1, 1.0, 0.6, 0.05, key="gun_conf")  # <-- changed default to 0.6
            process_video_or_webcam("Gun Detection", lambda frame: process_frame_with_yolov5_model(gun_model, frame, conf, draw_gun_boxes, alert_module_name="Weapon"))

    elif module == "🔥 Smoke & Fire Detection":
        st.subheader("🔥 Smoke & Fire Detection")
        st.info("Detects smoke and fire. An email and audio alert will be triggered instantly if a fire is detected.", icon="🚨")
        FIRE_MODEL_PATH = "best.pt"
        fire_model = load_yolo_model(FIRE_MODEL_PATH)
        if fire_model:
            st.success(f"✅ Fire Model '{FIRE_MODEL_PATH}' loaded.")
            conf = st.slider("Confidence", 0.1, 1.0, 0.6, 0.05, key="fire_conf")  # <-- changed default to 0.6
            process_video_or_webcam("Fire Detection", lambda frame: process_frame_with_model(fire_model, frame, conf, draw_fire_boxes, alert_module_name="Fire"))

    elif module == "🏎 Vehicle & Plate Recognition (ANPR)":
        st.subheader("🏎 Vehicle & License Plate Recognition")
        st.info("This module detects vehicles and reads their license plates.", icon="ℹ️")
        vehicle_model, lp_detector, ocr_reader = load_anpr_models()
        if all([vehicle_model, lp_detector, ocr_reader]):
            st.success("✅ ANPR models loaded successfully!")
            conf = st.slider("License Plate Confidence", 0.1, 1.0, 0.6, 0.05, key="anpr_conf")  # <-- changed default to 0.6
            process_video_or_webcam("ANPR", lambda frame: process_frame_for_anpr(frame, vehicle_model, lp_detector, ocr_reader, conf))

    st.markdown('</div>', unsafe_allow_html=True)


# --- FOOTER (Unchanged) ---
st.markdown("---")
st.markdown('<p class="footer">© 2024 Intelligent Vision Systems | Built with Streamlit</p>', unsafe_allow_html=True)
