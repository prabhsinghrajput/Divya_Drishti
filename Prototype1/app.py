import cv2
import torch
from ultralytics import YOLO
import pytesseract
import pyttsx3
from flask import Flask, Response, jsonify, render_template
import time
import threading
import subprocess
import platform
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# -----------------------------
# ⚙️ CONFIG
# -----------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

# Distance estimation parameters
KNOWN_WIDTH = 0.3        # Estimated average width of objects in meters
FOCAL_LENGTH = 500       # Camera focal length in pixels
RESIZE_WIDTH = 416       # Standard YOLO input size for better detection
FRAME_SKIP = 2          # Process every 2nd frame
SPEAK_COOLDOWN = 2      # Reduced cooldown for continuous direction updates
DIRECTIONAL_COOLDOWN = 1.5  # Special cooldown for directional mode

# Global state
camera_on = True
mode = "objects"        # "objects", "text", "distance", or "directional"
speech_on = True
last_speak_time = 0
frame_count = 0
directional_object_index = 0  # Track which object to announce next

# -----------------------------
# 🧭 DIRECTIONAL DETECTION
# -----------------------------
def get_object_position(x_center, frame_width):
    """Determine if object is on left, center, or right"""
    left_threshold = frame_width * 0.33   # Left third
    right_threshold = frame_width * 0.67  # Right third
    
    if x_center < left_threshold:
        return "left"
    elif x_center > right_threshold:
        return "right"
    else:
        return "center"

def get_object_distance_category(y_center, frame_height):
    """Determine if object is near, middle, or far based on vertical position"""
    near_threshold = frame_height * 0.7   # Bottom 30% = near
    
    if y_center > near_threshold:
        return "close"
    else:
        return "far"
# -----------------------------
# 📏 DISTANCE ESTIMATION
# -----------------------------
def estimate_distance(box_width_px):
    """Estimate distance to object based on bounding box width"""
    if box_width_px == 0:
        return None
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width_px
    return round(distance, 2)

def get_box_area(box):
    """Calculate bounding box area"""
    x1, y1, x2, y2 = box.xyxy[0]
    return (x2 - x1) * (y2 - y1)

# -----------------------------
# 🔊 IMPROVED SPEECH SYSTEM WITH PAUSES
# -----------------------------
def speak_single_directional(text):
    """Speak a single directional object with emphasis"""
    if not speech_on or not text.strip():
        return
    
    def speak():
        try:
            clean_text = text.strip().replace('"', '').replace("'", "")
            print(f"🧭 Speaking direction: '{clean_text}'")
            
            system = platform.system()
            
            if system == "Darwin":  # macOS
                try:
                    # Add emphasis for directional speech
                    subprocess.run(['say', '-r', '130', clean_text], check=True, timeout=6)
                    print(f"✅ Directional speech successful: {clean_text}")
                except Exception as e:
                    print(f"❌ Directional speech error: {e}")
                    
            else:  # Windows/Linux
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 130)  # Slower for clarity
                    engine.setProperty('volume', 1.0)
                    
                    voices = engine.getProperty('voices')
                    if voices:
                        engine.setProperty('voice', voices[0].id)
                    
                    engine.say(clean_text)
                    engine.runAndWait()
                    engine.stop()
                    del engine
                    
                    print(f"✅ Directional speech successful: {clean_text}")
                    
                except Exception as e:
                    print(f"❌ pyttsx3 directional error: {e}")
                    
        except Exception as e:
            print(f"❌ General directional speech error: {e}")
    
    speech_thread = threading.Thread(target=speak, daemon=True)
    speech_thread.start()
def speak_with_pauses(text_list):
    """Speak multiple items with pauses between them"""
    if not speech_on or not text_list:
        return
    
    def speak():
        try:
            system = platform.system()
            
            for i, text in enumerate(text_list):
                if not speech_on:  # Check if speech was disabled during speaking
                    break
                    
                clean_text = text.strip().replace('"', '').replace("'", "")
                print(f"🔊 Speaking item {i+1}/{len(text_list)}: '{clean_text}'")
                
                if system == "Darwin":  # macOS
                    try:
                        # Add natural pause with comma for multiple items
                        speech_text = clean_text
                        if i < len(text_list) - 1:
                            speech_text += "."  # Period adds pause
                        
                        subprocess.run(['say', speech_text], check=True, timeout=8)
                        print(f"✅ Speech successful for: {clean_text}")
                        
                        # Add extra pause between items (except for last item)
                        if i < len(text_list) - 1:
                            time.sleep(0.5)  # Half second pause
                            
                    except subprocess.TimeoutExpired:
                        print(f"⚠️ Speech timeout for: {clean_text}")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Speech error for {clean_text}: {e}")
                        
                else:  # Windows/Linux
                    try:
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 140)  # Slower for clarity
                        engine.setProperty('volume', 1.0)
                        
                        voices = engine.getProperty('voices')
                        if voices:
                            engine.setProperty('voice', voices[0].id)
                        
                        engine.say(clean_text)
                        engine.runAndWait()
                        engine.stop()
                        del engine
                        
                        print(f"✅ Speech successful for: {clean_text}")
                        
                        # Add pause between items
                        if i < len(text_list) - 1:
                            time.sleep(0.7)
                            
                    except Exception as e:
                        print(f"❌ pyttsx3 error for {clean_text}: {e}")
                        
        except Exception as e:
            print(f"❌ General speech error: {e}")
    
    # Run in separate thread to avoid blocking
    speech_thread = threading.Thread(target=speak, daemon=True)
    speech_thread.start()

def speak_async(text):
    """Single text speech function"""
    if not speech_on or not text.strip():
        return
    
    def speak():
        try:
            clean_text = text.strip().replace('"', '').replace("'", "")
            print(f"🔊 Speaking: '{clean_text}'")
            
            system = platform.system()
            
            if system == "Darwin":  # macOS
                try:
                    subprocess.run(['say', clean_text], check=True, timeout=8)
                    print("✅ Speech successful")
                except Exception as e:
                    print(f"❌ Speech error: {e}")
                    
            else:  # Windows/Linux
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.setProperty('volume', 1.0)
                    
                    voices = engine.getProperty('voices')
                    if voices:
                        engine.setProperty('voice', voices[0].id)
                    
                    engine.say(clean_text)
                    engine.runAndWait()
                    engine.stop()
                    del engine
                    
                    print("✅ Speech successful")
                    
                except Exception as e:
                    print(f"❌ pyttsx3 error: {e}")
                    
        except Exception as e:
            print(f"❌ General speech error: {e}")
    
    speech_thread = threading.Thread(target=speak, daemon=True)
    speech_thread.start()

# -----------------------------
# 🎥 VIDEO STREAM
# -----------------------------
@app.route("/video_feed")
def video_feed():
    def generate():
        global frame_count, last_speak_time
        
        cap = cv2.VideoCapture(2)  # Changed to 0 for built-in camera
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            global camera_on, mode, speech_on
            
            if not camera_on:
                _, buffer = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + 
                       buffer.tobytes() + b"\r\n")
                continue
            
            # Use original frame for display, resized for processing
            display_frame = frame.copy()
            
            # Create resized frame only for YOLO processing
            if mode in ["objects", "distance", "directional"]:
                h, w = frame.shape[:2]
                new_h = int(h * (RESIZE_WIDTH / w))
                frame_resized = cv2.resize(frame, (RESIZE_WIDTH, new_h))
                scale_x = w / RESIZE_WIDTH
                scale_y = h / new_h
            else:
                frame_resized = frame
                scale_x = scale_y = 1
            
            # Frame skipping for performance (except in text mode)
            frame_count += 1
            if mode not in ["text"] and frame_count % FRAME_SKIP != 0:
                _, buffer = cv2.imencode(".jpg", display_frame)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + 
                       buffer.tobytes() + b"\r\n")
                continue
            
            current_time = time.time()
            
            # Use different cooldowns based on mode
            if mode == "directional":
                speech_ready = speech_on and (current_time - last_speak_time) > DIRECTIONAL_COOLDOWN
            else:
                speech_ready = speech_on and (current_time - last_speak_time) > SPEAK_COOLDOWN
            
            if mode == "objects":
                # Standard object detection mode
                results = model(frame_resized, verbose=False)
                detected_names = set()
                
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        cls = int(box.cls[0])
                        conf = box.conf[0]
                        
                        if conf > 0.5:  # Confidence threshold
                            label = model.names[cls]
                            detected_names.add(label)
                            
                            # Scale coordinates back to original frame
                            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                            
                            # Draw bounding box and label
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if speech_ready and detected_names:
                    # Convert to list and sort for consistent ordering
                    objects_list = sorted(list(detected_names))[:3]  # Max 3 objects
                    print(f"🎯 Detected objects: {objects_list}")
                    
                    # Use paused speech for multiple objects
                    if len(objects_list) > 1:
                        speak_with_pauses(objects_list)
                        print(f"Speaking with pauses: {', '.join(objects_list)}")
                    else:
                        speak_async(objects_list[0])
                        print(f"Speaking single object: {objects_list[0]}")
                        
                    last_speak_time = current_time
                    
            elif mode == "distance":
                # Distance estimation mode - focus on largest object
                results = model(frame_resized, verbose=False)
                largest_box = None
                max_area = 0
                best_conf = 0
                
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        conf = box.conf[0]
                        if conf > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = box.xyxy[0]
                            area = (x2 - x1) * (y2 - y1)
                            if area > max_area:
                                max_area = area
                                largest_box = box
                                best_conf = conf
                
                if largest_box is not None:
                    x1, y1, x2, y2 = largest_box.xyxy[0]
                    cls = int(largest_box.cls[0])
                    label = model.names[cls]
                    
                    # Scale coordinates back to original frame
                    x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                    
                    # Calculate distance using original frame coordinates
                    box_width = x2 - x1
                    distance = estimate_distance(box_width)
                    
                    # Draw bounding box (thicker for distance mode)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    
                    # Display label, confidence, and distance
                    distance_text = f"{label} ({best_conf:.2f}): {distance}m" if distance else f"{label}: ?"
                    cv2.putText(display_frame, distance_text, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # Add large distance info at bottom
                    cv2.putText(display_frame, f"DISTANCE: {distance} meters", 
                              (20, display_frame.shape[0] - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                    
                    if speech_ready and distance:
                        distance_speech = f"{label} at {distance} meters"
                        speak_async(distance_speech)
                        last_speak_time = current_time
                        print(f"Speaking: {distance_speech}")
                        
            elif mode == "directional":
                # Directional object detection mode - announce one object at a time
                global directional_object_index
                results = model(frame_resized, verbose=False)
                directional_objects = []
                
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    frame_height, frame_width = display_frame.shape[:2]
                    
                    # Draw directional guides on screen
                    left_line = int(frame_width * 0.33)
                    right_line = int(frame_width * 0.67)
                    
                    # Draw vertical lines for left/center/right zones
                    cv2.line(display_frame, (left_line, 0), (left_line, frame_height), (100, 100, 100), 2)
                    cv2.line(display_frame, (right_line, 0), (right_line, frame_height), (100, 100, 100), 2)
                    
                    # Add zone labels with better visibility
                    cv2.putText(display_frame, "LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(display_frame, "CENTER", (int(frame_width*0.42), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(display_frame, "RIGHT", (int(frame_width*0.75), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                    
                    # Process all detected objects
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        cls = int(box.cls[0])
                        conf = box.conf[0]
                        
                        if conf > 0.5:  # Confidence threshold
                            label = model.names[cls]
                            
                            # Scale coordinates back to original frame
                            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                            
                            # Calculate object center
                            x_center = (x1 + x2) // 2
                            y_center = (y1 + y2) // 2
                            
                            # Determine position
                            position = get_object_position(x_center, frame_width)
                            distance_cat = get_object_distance_category(y_center, frame_height)
                            
                            # Create object info
                            obj_info = {
                                'label': label,
                                'position': position,
                                'distance_cat': distance_cat,
                                'coords': (x1, y1, x2, y2),
                                'center': (x_center, y_center)
                            }
                            directional_objects.append(obj_info)
                    
                    # Sort objects by position (left to right) for consistent ordering
                    directional_objects.sort(key=lambda obj: obj['center'][0])
                    
                    # Draw all objects with position colors
                    for i, obj in enumerate(directional_objects):
                        x1, y1, x2, y2 = obj['coords']
                        label = obj['label']
                        position = obj['position']
                        distance_cat = obj['distance_cat']
                        
                        # Color code by position
                        if position == "left":
                            color = (0, 255, 255)  # Yellow for left
                        elif position == "right":
                            color = (255, 0, 255)  # Magenta for right
                        else:
                            color = (0, 255, 0)    # Green for center
                        
                        # Highlight current object being announced
                        thickness = 4 if i == directional_object_index % len(directional_objects) else 2
                        
                        # Draw bounding box with position color
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Create position description
                        pos_text = f"{label} - {position.upper()}"
                        if distance_cat == "close":
                            pos_text += " (CLOSE)"
                        
                        # Add object number for tracking
                        cv2.putText(display_frame, f"{i+1}. {pos_text}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Announce one object at a time in cycle
                if speech_ready and directional_objects:
                    current_obj = directional_objects[directional_object_index % len(directional_objects)]
                    
                    # Create announcement text
                    announcement = f"{current_obj['label']} on your {current_obj['position']}"
                    if current_obj['distance_cat'] == "close":
                        announcement += " close to you"
                    
                    print(f"🧭 Announcing object {directional_object_index % len(directional_objects) + 1}/{len(directional_objects)}: {announcement}")
                    
                    # Speak the current object
                    speak_single_directional(announcement)
                    
                    # Move to next object for next announcement
                    directional_object_index += 1
                    last_speak_time = current_time
                    
                    # Add visual indicator for current announcement
                    cv2.putText(display_frame, f"ANNOUNCING: {announcement}", 
                              (20, display_frame.shape[0] - 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Show cycle progress
                    progress_text = f"Object {directional_object_index % len(directional_objects) + 1} of {len(directional_objects)}"
                    cv2.putText(display_frame, progress_text, 
                              (20, display_frame.shape[0] - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                        
            elif mode == "text":
                # Text recognition mode
                try:
                    text = pytesseract.image_to_string(display_frame)
                    clean_text = " ".join(text.strip().split())  # Clean whitespace
                    
                    if speech_ready and clean_text and len(clean_text) > 3:
                        # Limit text length for speech
                        speech_text = clean_text[:100] + "..." if len(clean_text) > 100 else clean_text
                        speak_async(speech_text)
                        last_speak_time = current_time
                        print(f"Speaking text: {speech_text}")
                        
                    # Display extracted text
                    lines = clean_text.split()
                    current_line = ""
                    y_pos = 40
                    
                    for word in lines[:15]:  # Show first 15 words
                        if len(current_line + word) < 25:
                            current_line += word + " "
                        else:
                            if current_line.strip():
                                cv2.putText(display_frame, current_line.strip(), (20, y_pos),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                y_pos += 25
                            current_line = word + " "
                            if y_pos > display_frame.shape[0] - 50:
                                break
                    
                    # Display remaining text
                    if current_line.strip() and y_pos <= display_frame.shape[0] - 50:
                        cv2.putText(display_frame, current_line.strip(), (20, y_pos),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                except Exception as e:
                    print(f"OCR Error: {e}")
                    cv2.putText(display_frame, "OCR Error - Check image quality", (20, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Add mode indicator
            mode_text = f"Mode: {mode.upper()}"
            cv2.putText(display_frame, mode_text, (20, 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Encode and yield frame
            _, buffer = cv2.imencode(".jpg", display_frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + 
                   buffer.tobytes() + b"\r\n")
        
        cap.release()
    
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# -----------------------------
# 🎛️ CONTROL ROUTES
# -----------------------------
@app.route("/pause", methods=["POST"])
def pause():
    global camera_on
    camera_on = False
    return jsonify({"status": "paused"})

@app.route("/resume", methods=["POST"])
def resume():
    global camera_on
    camera_on = True
    return jsonify({"status": "resumed"})

@app.route("/scan_text", methods=["POST"])
def scan_text():
    global mode
    mode = "text"
    return jsonify({"status": "text mode"})

@app.route("/scan_objects", methods=["POST"])
def scan_objects():
    global mode
    mode = "objects"
    return jsonify({"status": "object mode"})

@app.route("/distance_mode", methods=["POST"])
def distance_mode():
    global mode
    mode = "distance"
    return jsonify({"status": "distance mode"})

@app.route("/directional_mode", methods=["POST"])
def directional_mode():
    global mode
    mode = "directional"
    return jsonify({"status": "directional mode"})

@app.route("/test_speech", methods=["POST"])
def test_speech():
    """Test endpoint to check if speech is working"""
    # Test single directional announcement
    test_announcement = "person on your left close to you"
    print(f"🧪 Testing single directional speech: {test_announcement}")
    speak_single_directional(test_announcement)
    return jsonify({"status": "speech test sent", "announcement": test_announcement})

@app.route("/stop_speech", methods=["POST"])
def stop_speech():
    global speech_on
    speech_on = False
    print("🔇 Speech disabled")
    return jsonify({"status": "speech stopped"})

@app.route("/start_speech", methods=["POST"])
def start_speech():
    global speech_on
    speech_on = True
    print("🔊 Speech enabled")
    return jsonify({"status": "speech started"})

@app.route("/calibrate", methods=["POST"])
def calibrate():
    """Endpoint to adjust distance parameters"""
    from flask import request
    global KNOWN_WIDTH, FOCAL_LENGTH
    data = request.get_json()
    if 'known_width' in data:
        KNOWN_WIDTH = float(data['known_width'])
    if 'focal_length' in data:
        FOCAL_LENGTH = float(data['focal_length'])
    return jsonify({
        "status": "calibrated",
        "known_width": KNOWN_WIDTH,
        "focal_length": FOCAL_LENGTH
    })

@app.route("/stats")
def stats():
    return jsonify({
        "camera_on": camera_on,
        "mode": mode,
        "speech_on": speech_on,
        "known_width": KNOWN_WIDTH,
        "focal_length": FOCAL_LENGTH,
        "last_speak": last_speak_time
    })

@app.route("/")
def home():
    return render_template("index.html", cache_bust=str(int(time.time())))

# -----------------------------
# 🚀 MAIN
# -----------------------------
if __name__ == "__main__":
    print(f"🚀 Starting Enhanced Vision Assistant on device: {device}")
    print(f"📊 Modes available: objects, text, distance, directional")
    print(f"🔊 Speech system: {'macOS (say)' if platform.system() == 'Darwin' else 'pyttsx3'}")
    print(f"🎤 Test speech with: curl -X POST http://localhost:3000/test_speech")
    
    app.run(host="0.0.0.0", port=1000, debug=False, threaded=True)