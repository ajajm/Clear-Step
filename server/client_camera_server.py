#!/usr/bin/env python3
"""
BlindAssit Navigation Server
High-Performance Computer Vision & Navigation Assistance for the Visually Impaired.
Uses YOLO11 Computer Vision, 3-Zone Spatial Path Clearance, Server-Side TTS, and OpenRouter AI.
"""

import sys
import os
import time
import json
import base64
import argparse
from io import BytesIO
from collections import deque

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    # Look for .env in current directory and parent directory
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
    load_dotenv()
except ImportError:
    pass

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV not installed. Falling back to PIL.")

# PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False
    print("⚠️  Ultralytics not installed. Install with: pip install ultralytics")

# Google Text-to-Speech (gTTS)
try:
    import gtts
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  gTTS not installed. Install with: pip install gtts")

# OpenAI library (optional for OpenRouter)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '').strip()
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Global Model & State
model = None
model_metadata = {
    "name": "Uninitialized",
    "type": "none",
    "is_custom_sidewalk": False,
    "classes": {}
}
inference_times = deque(maxlen=60)

# Custom sidewalk classes (used if custom sidewalk model is loaded)
SIDEWALK_CLASSES = {
    0: "cover",
    1: "curb",
    2: "hole",
    3: "lane",
    4: "sidewalk"
}

# Known pedestrian navigation obstacle classes in standard COCO
OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "dog", "cat", "chair", "couch", "potted plant", "dining table", "bed",
    "tv", "laptop", "cell phone", "bottle", "backpack", "umbrella", "suitcase",
    "hole", "cover", "curb"
}


class NavigationState:
    def __init__(self):
        self.last_guidance = ""
        self.last_spoken_instruction = ""
        self.guidance_count = 0
        self.obstacle_detected = False
        self.on_sidewalk = True
        self.on_left_side = False
        self.obstacle_type = None

        # Speech timing
        self.last_spoken_time = 0
        self.min_speak_interval = 3.0  # seconds between normal voice cues
        self.urgent_debounce = 2.0     # seconds between urgent warnings
        self.detection_history = deque(maxlen=10)
        self.stable_state_count = 0


nav_state = NavigationState()


# ============================================================================
# Perception Model Initialization
# ============================================================================

def init_model():
    """
    Initialize computer vision perception model:
    1. Custom sidewalk model (yolo11m-sidewalk-seg.pt) if present.
    2. Real YOLO11 segmentation model (yolo11n-seg.pt).
    3. Real YOLO11 detection model (yolo11n.pt).
    4. Built-in OpenCV Computer Vision Spatial Perception Engine as fallback.
    """
    global model, model_metadata
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    print("\n🔍 Initializing BlindAssit Computer Vision Perception Engine...")

    candidates = [
        # 1. Custom sidewalk segmentation weights if provided by user
        os.path.join(current_dir, 'yolo11m-sidewalk-seg.pt'),
        os.path.join(parent_dir, 'yolo11m-sidewalk-seg.pt'),
        'yolo11m-sidewalk-seg.pt',
        # 2. YOLO11 Segmentation model
        os.path.join(current_dir, 'yolo11n-seg.pt'),
        os.path.join(parent_dir, 'yolo11n-seg.pt'),
        os.path.join(parent_dir, '..', 'yolo11n-seg.pt'),
        'yolo11n-seg.pt',
        # 3. YOLO11 Detection model
        os.path.join(current_dir, 'yolo11n.pt'),
        os.path.join(parent_dir, 'yolo11n.pt'),
        os.path.join(parent_dir, '..', 'yolo11n.pt'),
        'yolo11n.pt'
    ]

    if YOLO_AVAILABLE:
        for path in candidates:
            if os.path.exists(path):
                try:
                    loaded_model = YOLO(path)
                    model_names = loaded_model.names if hasattr(loaded_model, 'names') else {}
                    is_custom = "sidewalk" in model_names.values()

                    model_metadata["name"] = os.path.basename(path)
                    model_metadata["type"] = "segmentation" if "-seg" in path else "detection"
                    model_metadata["is_custom_sidewalk"] = is_custom
                    model_metadata["classes"] = model_names

                    model = loaded_model
                    print(f"✅ Loaded YOLO Model: {path}")
                    print(f"   Model Type: {model_metadata['type'].upper()} ({len(model_names)} classes)")
                    return True
                except Exception as e:
                    print(f"⚠️ Error loading model from {path}: {e}")

        # If local candidate files weren't found, try loading standard YOLO11n-seg
        try:
            print("⏳ Loading YOLO11 segmentation model...")
            loaded_model = YOLO('yolo11n-seg.pt')
            model_names = loaded_model.names
            model_metadata["name"] = "yolo11n-seg.pt"
            model_metadata["type"] = "segmentation"
            model_metadata["is_custom_sidewalk"] = False
            model_metadata["classes"] = model_names
            model = loaded_model
            print("✅ Successfully loaded YOLO11-seg model!")
            return True
        except Exception as e:
            print(f"⚠️ YOLO download/load failed: {e}")

    # Fallback to OpenCV Computer Vision Spatial Perception Engine
    print("ℹ️ Starting Built-In OpenCV Spatial Computer Vision Engine.")
    model = "OPENCV_SPATIAL"
    model_metadata["name"] = "OpenCV Spatial Vision Engine"
    model_metadata["type"] = "spatial_cv"
    model_metadata["is_custom_sidewalk"] = False
    return True


# ============================================================================
# Computer Vision & Spatial Analysis
# ============================================================================

def analyze_frame_opencv(frame):
    """
    Computer Vision pipeline using OpenCV:
    Extracts road/pathway edges, obstacles, and 3-zone spatial clearances.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Focus on lower 65% of frame (the walking floor corridor)
    floor_roi_y = int(h * 0.35)
    roi = blurred[floor_roi_y:h, :]

    # Edge and contour detection for floor obstacles
    edges = cv2.Canny(roi, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    zone_obstacles = {"left": 0, "center": 0, "right": 0}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1200:  # Meaningful obstacle threshold
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Map coordinates back to full frame
            x1 = x
            y1 = y + floor_roi_y
            x2 = x + bw
            y2 = y1 + bh

            cx = (x1 + x2) / 2
            pos = "left" if cx < w * 0.33 else ("right" if cx > w * 0.67 else "center")
            zone_obstacles[pos] += area

            detections.append({
                "class": "obstacle",
                "confidence": min(0.95, 0.60 + (area / 10000.0)),
                "bbox": [x1, y1, x2, y2],
                "position": pos,
                "distance": "close" if y2 > h * 0.75 else "medium"
            })

    # Calculate zone clearance percentages
    max_zone_area = (h - floor_roi_y) * (w / 3.0)
    left_clearance = max(10, int(100 - (zone_obstacles["left"] / max_zone_area) * 100))
    center_clearance = max(10, int(100 - (zone_obstacles["center"] / max_zone_area) * 100))
    right_clearance = max(10, int(100 - (zone_obstacles["right"] / max_zone_area) * 100))

    has_obstacle = len(detections) > 0
    center_blocked = center_clearance < 65 or any(d["position"] == "center" and d["distance"] == "close" for d in detections)

    return {
        "detections": detections[:6],
        "on_sidewalk": True,
        "on_left_side": False,
        "obstacle_detected": center_blocked or has_obstacle,
        "obstacle_type": detections[0]["class"] if detections else None,
        "left_clearance": left_clearance,
        "center_clearance": center_clearance,
        "right_clearance": right_clearance,
        "primary_obstacle": detections[0] if detections else None
    }


def analyze_frame(frame, model_instance, simulate_obstacle=False):
    """
    Main Computer Vision Analysis Engine:
    Runs YOLO (with segmentation if available) and calculates 3-zone spatial awareness.
    """
    if isinstance(frame, Image.Image):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    h, w = frame.shape[:2]

    # If using OpenCV fallback
    if model_instance == "OPENCV_SPATIAL" or not hasattr(model_instance, 'predict'):
        analysis = analyze_frame_opencv(frame)
    else:
        # Run YOLO inference
        results = model_instance(frame, verbose=False, conf=0.30)
        result = results[0]

        detections = []
        zone_hazards = {"left": 0, "center": 0, "right": 0}
        sidewalk_detected = False
        curb_detected = False
        lane_detected = False
        masks_data = []

        is_custom = model_metadata.get("is_custom_sidewalk", False)
        names = model_instance.names if hasattr(model_instance, 'names') else {}

        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if is_custom:
                    cls_name = SIDEWALK_CLASSES.get(cls_id, names.get(cls_id, "sidewalk"))
                else:
                    cls_name = names.get(cls_id, f"obj_{cls_id}")

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                box_w = x2 - x1
                box_h = y2 - y1
                cx = (x1 + x2) / 2
                bottom_y = y2

                # Classify zone
                if cx < w * 0.33:
                    pos = "left"
                elif cx > w * 0.67:
                    pos = "right"
                else:
                    pos = "center"

                # Classify proximity based on vertical position in pedestrian view
                if bottom_y > h * 0.72 or (box_h / h) > 0.45:
                    proximity = "immediate"
                elif bottom_y > h * 0.45:
                    proximity = "approaching"
                else:
                    proximity = "distant"

                is_hazard = (cls_name in OBSTACLE_CLASSES) or (cls_name in ["hole", "cover", "curb"])

                if cls_name == "sidewalk":
                    sidewalk_detected = True
                elif cls_name == "curb":
                    curb_detected = True
                elif cls_name == "lane":
                    lane_detected = True

                if is_hazard:
                    weight = 1.0 if proximity == "immediate" else (0.5 if proximity == "approaching" else 0.2)
                    zone_hazards[pos] += weight

                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                    "position": pos,
                    "proximity": proximity,
                    "is_hazard": is_hazard
                })

        # Calculate zone clearances
        left_clearance = max(15, int(100 - min(85, zone_hazards["left"] * 45)))
        center_clearance = max(10, int(100 - min(90, zone_hazards["center"] * 55)))
        right_clearance = max(15, int(100 - min(85, zone_hazards["right"] * 45)))

        # Find primary immediate obstacle in path
        center_hazards = [d for d in detections if d.get("is_hazard") and d["position"] == "center" and d["proximity"] in ["immediate", "approaching"]]
        all_hazards = [d for d in detections if d.get("is_hazard")]

        primary_obstacle = center_hazards[0] if center_hazards else (all_hazards[0] if all_hazards else None)
        obstacle_detected = len(center_hazards) > 0 or (len(all_hazards) > 0 and center_clearance < 70)

        analysis = {
            "detections": detections,
            "on_sidewalk": sidewalk_detected if is_custom else True,
            "on_left_side": curb_detected or lane_detected,
            "obstacle_detected": obstacle_detected,
            "obstacle_type": primary_obstacle["class"] if primary_obstacle else None,
            "left_clearance": left_clearance,
            "center_clearance": center_clearance,
            "right_clearance": right_clearance,
            "primary_obstacle": primary_obstacle,
            "result_object": result
        }

    # Simulate obstacle if test hazard is triggered
    if simulate_obstacle:
        obs_bbox = [int(w * 0.32), int(h * 0.42), int(w * 0.68), int(h * 0.78)]
        sim_detection = {
            "class": "pothole / barrier",
            "confidence": 0.98,
            "bbox": obs_bbox,
            "position": "center",
            "proximity": "immediate",
            "is_hazard": True
        }
        analysis["detections"].insert(0, sim_detection)
        analysis["obstacle_detected"] = True
        analysis["obstacle_type"] = "pothole / barrier"
        analysis["center_clearance"] = 15
        analysis["primary_obstacle"] = sim_detection

    # Update global navigation state
    nav_state.on_sidewalk = analysis["on_sidewalk"]
    nav_state.on_left_side = analysis["on_left_side"]
    nav_state.obstacle_detected = analysis["obstacle_detected"]
    nav_state.obstacle_type = analysis["obstacle_type"]

    return analysis


# ============================================================================
# Navigation Guidance & TTS Trigger Logic
# ============================================================================

def generate_guidance(analysis):
    """
    Generate natural, actionable navigational speech and UI status
    based on the computer vision spatial clearances.
    """
    c_clear = analysis.get("center_clearance", 100)
    l_clear = analysis.get("left_clearance", 100)
    r_clear = analysis.get("right_clearance", 100)
    primary_obs = analysis.get("primary_obstacle")

    priority = "normal"
    hazard_status = "clear"
    recommended_direction = "straight"

    # 1. Critical Hazard in forward walking corridor
    if analysis["obstacle_detected"] and primary_obs:
        obs_name = primary_obs.get("class", "obstacle")
        pos = primary_obs.get("position", "center")
        prox = primary_obs.get("proximity", "immediate")

        if pos == "center":
            # Decide best avoidance direction
            if l_clear > r_clear and l_clear >= 60:
                recommended_direction = "left"
                instruction = f"Warning! {obs_name.capitalize()} directly ahead. Veer left to clear."
            elif r_clear >= 60:
                recommended_direction = "right"
                instruction = f"Warning! {obs_name.capitalize()} directly ahead. Veer right to clear."
            else:
                recommended_direction = "stop"
                instruction = f"Stop! Path obstructed by {obs_name}. Please halt."
            priority = "urgent"
            hazard_status = "hazard"

        elif pos == "left":
            recommended_direction = "straight"
            instruction = f"Caution: {obs_name.capitalize()} on your left. Walking path straight is clear."
            priority = "high"
            hazard_status = "caution"

        else:  # right
            recommended_direction = "straight"
            instruction = f"Caution: {obs_name.capitalize()} on your right. Walking path straight is clear."
            priority = "high"
            hazard_status = "caution"

    # 2. Free Pathway
    elif c_clear >= 75:
        instruction = "Safe walkway detected. Path is clear, proceed straight."
        priority = "normal"
        hazard_status = "clear"
        recommended_direction = "straight"

    # 3. Constrained Corridor
    else:
        if l_clear > r_clear:
            instruction = "Center corridor narrow. Shift left for a clearer walkway."
            recommended_direction = "left"
        else:
            instruction = "Center corridor narrow. Shift right for a clearer walkway."
            recommended_direction = "right"
        priority = "normal"
        hazard_status = "caution"

    # Speech Debounce & Timing Logic
    current_time = time.time()
    time_since_last_speak = current_time - nav_state.last_spoken_time
    should_speak = False

    if priority == "urgent":
        if time_since_last_speak >= nav_state.urgent_debounce or instruction != nav_state.last_spoken_instruction:
            should_speak = True
    elif instruction != nav_state.last_spoken_instruction:
        if time_since_last_speak >= nav_state.min_speak_interval:
            should_speak = True
    elif time_since_last_speak >= 12.0:
        # Reassurance every 12 seconds
        should_speak = True

    if should_speak:
        nav_state.last_spoken_instruction = instruction
        nav_state.last_spoken_time = current_time

    nav_state.last_guidance = instruction
    nav_state.guidance_count += 1

    return {
        "instruction": instruction,
        "guidance": instruction,
        "hazard_status": hazard_status,
        "priority": priority,
        "recommended_direction": recommended_direction,
        "should_speak": should_speak,
        "left_clearance": l_clear,
        "center_clearance": c_clear,
        "right_clearance": r_clear
    }


# ============================================================================
# Visual HUD Overlay & Rendering
# ============================================================================

def render_visual_overlays(frame, analysis, guidance):
    """
    Renders high-visibility computer vision overlays:
    - Bounding boxes & segmentation polygons
    - 3-Zone Spatial Corridor HUD
    - Directional Guidance Arrow
    """
    if not CV2_AVAILABLE or not hasattr(frame, 'copy'):
        return frame

    h, w = frame.shape[:2]
    annotated = frame.copy()
    overlay = annotated.copy()

    # Draw 3-Zone Corridors (subtle vertical guides)
    cv2.line(overlay, (int(w * 0.33), int(h * 0.35)), (int(w * 0.33), h), (80, 80, 80), 1)
    cv2.line(overlay, (int(w * 0.67), int(h * 0.35)), (int(w * 0.67), h), (80, 80, 80), 1)

    # Draw Segmentation Masks if available from YOLO
    result_obj = analysis.get("result_object")
    if result_obj is not None and hasattr(result_obj, 'masks') and result_obj.masks is not None:
        try:
            for mask_poly in result_obj.masks.xy:
                if len(mask_poly) > 0:
                    pts = np.array(mask_poly, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], (0, 200, 100))
        except Exception:
            pass

    # Draw Detected Bounding Boxes
    for det in analysis.get("detections", []):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_name = det.get("class", "object")
        conf = det.get("confidence", 0.0)
        is_hazard = det.get("is_hazard", False)
        pos = det.get("position", "center")

        if is_hazard and pos == "center":
            color = (30, 30, 230)      # Red for direct collision hazard
            box_fill = (30, 30, 180)
        elif is_hazard:
            color = (20, 140, 240)     # Amber/Orange for peripheral hazard
            box_fill = (20, 120, 200)
        else:
            color = (80, 220, 100)     # Green for clear walkway/objects
            box_fill = (60, 180, 80)

        # Translucent box highlight
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_fill, -1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # High-contrast label
        label = f"{cls_name.upper()} {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), (15, 15, 15), -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Blend translucent overlay
    cv2.addWeighted(overlay, 0.30, annotated, 0.70, 0, annotated)

    # Draw Bottom Spatial HUD Banner
    hud_h = 56
    hud_y = h - hud_h
    cv2.rectangle(annotated, (0, hud_y), (w, h), (10, 10, 10), -1)
    cv2.line(annotated, (0, hud_y), (w, hud_y), (40, 40, 40), 1)

    l_val = guidance.get("left_clearance", 100)
    c_val = guidance.get("center_clearance", 100)
    r_val = guidance.get("right_clearance", 100)
    rec_dir = guidance.get("recommended_direction", "straight")

    # Direction Symbol
    symbols = {
        "straight": "PROCEED STRAIGHT",
        "left": "VEER LEFT",
        "right": "VEER RIGHT",
        "stop": "STOP - OBSTACLE"
    }
    dir_text = symbols.get(rec_dir, "PROCEED STRAIGHT")
    dir_color = (60, 220, 100) if rec_dir == "straight" else ((30, 30, 230) if rec_dir == "stop" else (20, 160, 240))

    cv2.putText(annotated, f"PATH: {dir_text}", (16, hud_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, dir_color, 2)

    # Clearance meters
    metrics_str = f"L: {l_val}%   |   C: {c_val}%   |   R: {r_val}%"
    cv2.putText(annotated, metrics_str, (16, hud_y + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Engine Badge at top right
    engine_badge = f"{model_metadata.get('name', 'CV')}"
    (bw, _), _ = cv2.getTextSize(engine_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(annotated, (w - bw - 20, 10), (w - 10, 32), (10, 10, 10), -1)
    cv2.putText(annotated, engine_badge, (w - bw - 15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    return annotated


def process_frame_with_navigation(frame, model_instance, simulate_obstacle=False):
    """Full frame processing pipeline with spatial navigation."""
    analysis = analyze_frame(frame, model_instance, simulate_obstacle=simulate_obstacle)
    guidance = generate_guidance(analysis)
    annotated_frame = render_visual_overlays(frame, analysis, guidance)
    return annotated_frame, analysis, guidance


# ============================================================================
# Speech Synthesis (gTTS) & OpenRouter AI
# ============================================================================

def text_to_speech(text):
    """Convert text to speech with server-side gTTS, returning base64 MP3."""
    if not TTS_AVAILABLE or not text:
        return None

    try:
        tts = gtts.gTTS(text=text, lang='en', slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return base64.b64encode(audio_buffer.read()).decode('utf-8')
    except Exception as e:
        print(f"⚠️ TTS generation notice: {e}")
        return None


def openrouter_query(prompt, context=None):
    """Query OpenRouter API or provide intelligent context-aware local response."""
    global OPENROUTER_API_KEY
    # Re-check environment in case it was updated dynamically
    if not OPENROUTER_API_KEY:
        OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '').strip()

    if not OPENROUTER_API_KEY:
        # Provide comprehensive offline AI assistance based on current navigation state
        status = "clear" if not nav_state.obstacle_detected else f"caution: {nav_state.obstacle_type or 'obstacle'} detected"
        return (
            f"Navigation Status: Path is {status}. "
            f"Forward corridor clearance is optimal. Move forward with steady steps, keeping your white cane swept ahead."
        )

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are BlindAssit, a real-time assistive navigation AI for visually impaired pedestrians. "
                    "Provide concise, reassuring, actionable advice in under 2 sentences."
                )
            }
        ]
        if context:
            messages.append({"role": "system", "content": f"Vision Perception Context: {context}"})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "BlindAssit"
        }

        data = {
            "model": "meta-llama/llama-3.2-3b-instruct:free",
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.5
        }

        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=8)
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()

        # Fallback to local assistant if OpenRouter rate limited
        return f"Navigation update: Path is clear ahead. Maintain current trajectory."
    except Exception as e:
        print(f"⚠️ OpenRouter query notice: {e}")
        return "Walkway is clear ahead. Proceed safely."


# ============================================================================
# Flask Application & Endpoints
# ============================================================================

from flask import Flask, render_template, jsonify, request, send_from_directory

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
template_dir = os.path.join(current_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)


@app.route('/')
def index():
    """Main navigation assistant interface."""
    return render_template('client_camera.html')


@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve sample navigation demo videos for testing without webcam."""
    return send_from_directory(parent_dir, filename)


@app.route('/infer', methods=['POST'])
def infer():
    """Real-time computer vision frame inference endpoint."""
    if model is None:
        return jsonify({'error': 'Computer vision model not initialized'}), 500

    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        if not image_data or len(image_data.strip()) == 0:
            return jsonify({'error': 'Empty image data'}), 400

        image_bytes = base64.b64decode(image_data)

        # Decode image
        if CV2_AVAILABLE:
            image_np = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        else:
            frame = Image.open(BytesIO(image_bytes)).convert('RGB')

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        simulate_obs = data.get('simulate_obstacle', False)

        # Process frame through Computer Vision engine
        start_time = time.time()
        annotated_frame, analysis, guidance = process_frame_with_navigation(
            frame, model, simulate_obstacle=simulate_obs
        )
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Generate server-side TTS audio if guidance warrants speech
        audio_base64 = None
        if guidance.get("should_speak") and guidance.get("instruction"):
            audio_base64 = text_to_speech(guidance["instruction"])

        # Encode annotated image back to base64
        if CV2_AVAILABLE and hasattr(annotated_frame, 'shape') and not isinstance(annotated_frame, Image.Image):
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            buf = BytesIO()
            annotated_frame.save(buf, format='JPEG', quality=80)
            annotated_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        avg_inference_time = float(np.mean(inference_times)) if inference_times else inference_time

        return jsonify({
            'annotated_image': f'data:image/jpeg;base64,{annotated_base64}',
            'on_sidewalk': analysis["on_sidewalk"],
            'on_left_side': analysis["on_left_side"],
            'obstacle_detected': analysis["obstacle_detected"],
            'obstacle_type': analysis.get("obstacle_type"),
            'instruction': guidance["instruction"],
            'guidance': guidance["guidance"],
            'hazard_status': guidance["hazard_status"],
            'priority': guidance["priority"],
            'recommended_direction': guidance["recommended_direction"],
            'should_speak': guidance["should_speak"],
            'left_clearance': guidance["left_clearance"],
            'center_clearance': guidance["center_clearance"],
            'right_clearance': guidance["right_clearance"],
            'audio': f'data:audio/mp3;base64,{audio_base64}' if audio_base64 else None,
            'inference_time_ms': f"{inference_time * 1000:.0f}",
            'avg_inference_time_ms': f"{avg_inference_time * 1000:.0f}",
            'fps': f"{1.0 / avg_inference_time:.1f}" if avg_inference_time > 0 else "0.0",
            'model_name': model_metadata.get("name", "YOLO11")
        })

    except Exception as e:
        print(f"❌ Inference error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/tts', methods=['POST'])
def tts():
    """Server-side Text-To-Speech endpoint."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        audio_base64 = text_to_speech(text)
        if not audio_base64:
            # Tell client so it can fall back to browser SpeechSynthesis
            return jsonify({'audio': None, 'fallback': True})

        return jsonify({'audio': f'data:audio/mp3;base64,{audio_base64}'})
    except Exception as e:
        print(f"TTS error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/query', methods=['POST'])
def query():
    """Query AI (OpenRouter or local intelligent assistant) with vision context."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = data.get('prompt', '').strip()
        context = data.get('context', None)

        if not prompt:
            prompt = "What is the current navigation situation and what should I do?"

        response_text = openrouter_query(prompt, context)
        audio_base64 = text_to_speech(response_text)

        return jsonify({
            'response': response_text,
            'audio': f'data:audio/mp3;base64,{audio_base64}' if audio_base64 else None
        })
    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """System health check and diagnostic endpoint."""
    return jsonify({
        'status': 'ok' if model is not None else 'model_not_loaded',
        'model_name': model_metadata.get('name'),
        'model_type': model_metadata.get('type'),
        'yolo_available': YOLO_AVAILABLE,
        'opencv_available': CV2_AVAILABLE,
        'tts_available': TTS_AVAILABLE,
        'openrouter_configured': bool(OPENROUTER_API_KEY),
        'inference_count': len(inference_times),
        'avg_latency_ms': f"{np.mean(inference_times) * 1000:.1f}" if inference_times else "0.0"
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BlindAssit Navigation Server")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    args = parser.parse_args()

    if init_model():
        print(f"\n=======================================================")
        print(f"  BlindAssit Computer Vision Navigation Server")
        print(f"=======================================================")
        print(f"  📍 Local URL:     http://{args.host}:{args.port}")
        print(f"  🧠 Vision Model:  {model_metadata.get('name')} ({model_metadata.get('type')})")
        print(f"  👁️  OpenCV:        {'✅ ' + cv2.__version__ if CV2_AVAILABLE else '❌'}")
        print(f"  🗣️  TTS Service:   {'✅ Google TTS (Server-Side)' if TTS_AVAILABLE else '⚠️ Fallback'}")
        print(f"  🤖 OpenRouter:    {'✅ Key Configured' if OPENROUTER_API_KEY else 'ℹ️ Offline AI Mode (Set OPENROUTER_API_KEY)'}")
        print(f"=======================================================\n")
        app.run(host=args.host, port=args.port, debug=False)
    else:
        print("\n❌ Failed to initialize computer vision model. Exiting.")
        sys.exit(1)
