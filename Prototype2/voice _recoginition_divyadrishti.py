import cv2
import torch
from ultralytics import YOLO
import pytesseract
import pyttsx3
from flask import Flask, Response, jsonify, request
import time
import threading
import subprocess
import platform
import numpy as np
from collections import deque
import logging

# Suppress verbose logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

app = Flask(__name__)

# Initialize model
try:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO("yolov8n.pt").to(device)
    print(f"Model loaded on device: {device}")
except Exception as e:
    print(f"Model loading error: {e}")
    device = "cpu"
    model = YOLO("yolov8n.pt")

# Voice recognition setup
VOICE_AVAILABLE = False
try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
    print("Voice recognition available")
except ImportError:
    print("Voice recognition not available")

# Configuration parameters
KNOWN_WIDTH = 0.6
FOCAL_LENGTH = 615
RESIZE_WIDTH = 416
FRAME_SKIP = 2
SPEAK_COOLDOWN = 4.0
DIRECTIONAL_COOLDOWN = 2.0
TEXT_COOLDOWN = 8.0
VOICE_COMMAND_COOLDOWN = 3.0

# Anti-jitter parameters
CONFIDENCE_THRESHOLD = 0.6
MIN_BOX_AREA = 1200
POSITION_SMOOTHING_FRAMES = 7
MIN_POSITION_CONFIDENCE = 5

# Global state
camera_on = True
mode = "objects"
speech_on = True
voice_commands_on = False
last_speak_time = 0
last_voice_command_time = 0
frame_count = 0
cap = None
directional_object_index = 0

# Tracking state
position_history = deque(maxlen=POSITION_SMOOTHING_FRAMES)
last_stable_position = None
last_spoken_objects = []
processing_lock = threading.Lock()
voice_listener = None

class EnhancedSpeechEngine:
    def __init__(self):
        self.engine = None
        self.is_speaking = False
        self.speech_queue = deque(maxlen=3)
        self.speech_lock = threading.Lock()
        self._initialize_engine()
        self._start_speech_worker()
    
    def _initialize_engine(self):
        try:
            if platform.system() != "Darwin":
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 130)
                self.engine.setProperty('volume', 1.0)
                
                voices = self.engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if any(keyword in voice.name.lower() for keyword in ['zira', 'hazel', 'female']):
                            self.engine.setProperty('voice', voice.id)
                            break
                
                print("Enhanced speech engine initialized")
        except Exception as e:
            print(f"Speech engine error: {e}")
    
    def _start_speech_worker(self):
        def worker():
            while True:
                try:
                    if self.speech_queue and not self.is_speaking:
                        with self.speech_lock:
                            if self.speech_queue:
                                text, priority = self.speech_queue.popleft()
                                self._speak_now(text)
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Speech worker error: {e}")
                    time.sleep(1)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def speak(self, text, priority=False):
        if not speech_on and not priority:
            return
        
        clean_text = str(text).strip()[:80]
        if not clean_text:
            return
        
        with self.speech_lock:
            if priority:
                self.speech_queue.clear()
            self.speech_queue.append((clean_text, priority))
    
    def speak_directional(self, text):
        if not speech_on:
            return
        
        clean_text = str(text).strip()
        if not clean_text:
            return
        
        with self.speech_lock:
            self.speech_queue.clear()
            self.speech_queue.append((clean_text, True))
    
    def _speak_now(self, text):
        self.is_speaking = True
        try:
            print(f"SPEAKING: '{text}'")
            
            if platform.system() == "Darwin":
                subprocess.run(['say', '-v', 'Samantha', '-r', '140', text], 
                             timeout=30, check=True)
            else:
                if self.engine:
                    self.engine.setProperty('rate', 130)
                    self.engine.say(text)
                    self.engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            self.is_speaking = False
            time.sleep(0.3)

class RobustVoiceListener:
    def __init__(self):
        self.active = False
        self.recognizer = None
        self.microphone = None
        self.error_count = 0
        self.max_errors = 10
        
        if VOICE_AVAILABLE:
            self._setup_voice()
    
    def _setup_voice(self):
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 250
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 1.5
            self.recognizer.phrase_threshold = 0.3
            
            for i in range(3):
                try:
                    self.microphone = sr.Microphone(device_index=i if i > 0 else None)
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    print(f"Voice initialized with microphone {i}")
                    break
                except Exception:
                    continue
            
        except Exception as e:
            print(f"Voice setup error: {e}")
    
    def start(self):
        if not VOICE_AVAILABLE or not self.recognizer or not self.microphone:
            return False
        
        try:
            self.active = True
            self.error_count = 0
            threading.Thread(target=self._listen_loop, daemon=True).start()
            return True
        except Exception as e:
            print(f"Voice start error: {e}")
            return False
    
    def stop(self):
        self.active = False
    
    def _listen_loop(self):
        print("Voice listening started - say 'enable voice' to activate")
        
        while self.active and self.error_count < self.max_errors:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                text = self.recognizer.recognize_google(audio).lower().strip()
                
                if text and len(text) > 1:
                    current_time = time.time()
                    global last_voice_command_time
                    
                    if current_time - last_voice_command_time > VOICE_COMMAND_COOLDOWN:
                        print(f"VOICE COMMAND: {text}")
                        self._process_command(text)
                        last_voice_command_time = current_time
                        self.error_count = 0
                
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                self.error_count += 1
                print(f"Voice service error: {e}")
                time.sleep(2)
            except Exception as e:
                self.error_count += 1
                print(f"Voice error: {e}")
                time.sleep(1)
        
        print("Voice listening stopped")
    
    def _process_command(self, text):
        global mode, camera_on, speech_on, voice_commands_on, directional_object_index
        
        # Priority commands
        if any(phrase in text for phrase in ["enable voice", "voice on", "activate voice"]):
            voice_commands_on = True
            speech_engine.speak("Voice commands enabled", priority=True)
            return
        
        if "help" in text:
            speech_engine.speak("Say objects, distance, directional, text, start, stop, speech on, or speech off", priority=True)
            return
        
        if not voice_commands_on:
            return
        
        # Mode commands
        if "objects" in text or "object" in text:
            mode = "objects"
            speech_engine.speak("Objects mode")
        elif "distance" in text or "measure" in text:
            mode = "distance"
            speech_engine.speak("Distance mode")
        elif any(word in text for word in ["directional", "direction", "navigate", "navigation"]):
            mode = "directional"
            directional_object_index = 0
            speech_engine.speak("Navigation mode")
        elif "text" in text or "read" in text:
            mode = "text"
            speech_engine.speak("Text mode")
        elif "start" in text or "begin" in text:
            camera_on = True
            speech_engine.speak("Started")
        elif "stop" in text or "pause" in text:
            camera_on = False
            speech_engine.speak("Stopped")
        elif "speech on" in text or "talk" in text:
            speech_on = True
            speech_engine.speak("Speech enabled")
        elif "speech off" in text or "quiet" in text:
            speech_on = False
            print("Speech disabled")
        elif "voice off" in text or "disable voice" in text:
            voice_commands_on = False
            speech_engine.speak("Voice disabled")

# Initialize components
speech_engine = EnhancedSpeechEngine()
if VOICE_AVAILABLE:
    voice_listener = RobustVoiceListener()

def get_stable_detections(detections):
    stable_detections = []
    
    for detection in detections:
        x1, y1, x2, y2, confidence, class_name = detection
        
        box_area = (x2 - x1) * (y2 - y1)
        box_ratio = (x2 - x1) / max(y2 - y1, 1)
        
        if (confidence > CONFIDENCE_THRESHOLD and 
            box_area > MIN_BOX_AREA and
            0.15 < box_ratio < 8.0):
            stable_detections.append(detection)
    
    return stable_detections

def get_object_position_stable(x_center, frame_width):
    # Conservative thresholds for stability
    left_threshold = frame_width * 0.35
    right_threshold = frame_width * 0.65
    
    if x_center < left_threshold:
        current_pos = "left"
    elif x_center > right_threshold:
        current_pos = "right"
    else:
        current_pos = "center"
    
    position_history.append(current_pos)
    
    if len(position_history) >= MIN_POSITION_CONFIDENCE:
        recent_positions = list(position_history)[-MIN_POSITION_CONFIDENCE:]
        position_counts = {}
        for pos in recent_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        most_common = max(position_counts, key=position_counts.get)
        
        if position_counts[most_common] >= (MIN_POSITION_CONFIDENCE - 1):
            global last_stable_position
            if most_common != last_stable_position:
                print(f"Position confirmed: {most_common}")
                last_stable_position = most_common
    
    return last_stable_position or "center"

def get_object_distance_category(y_center, frame_height):
    near_threshold = frame_height * 0.65
    return "close" if y_center > near_threshold else "far"

def initialize_camera():
    global cap
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        cap.set(cv2.CAP_PROP_FPS, 20)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Camera initialized with enhanced settings")
        return True
    except Exception as e:
        print(f"Camera initialization error: {e}")
        return False

def process_frame_enhanced(frame):
    global frame_count, last_speak_time
    
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        return frame
    
    if not processing_lock.acquire(blocking=False):
        return frame
    
    try:
        current_time = time.time()
        
        if mode == "objects":
            return process_objects_enhanced(frame, current_time)
        elif mode == "distance":
            return process_distance_enhanced(frame, current_time)
        elif mode == "directional":
            return process_directional_enhanced(frame, current_time)
        elif mode == "text":
            return process_text_enhanced(frame, current_time)
    except Exception as e:
        print(f"Processing error: {e}")
    finally:
        processing_lock.release()
    
    return frame

def process_objects_enhanced(frame, current_time):
    global last_speak_time, last_spoken_objects
    
    height, width = frame.shape[:2]
    small_frame = cv2.resize(frame, (RESIZE_WIDTH, int(RESIZE_WIDTH * height / width)))
    
    try:
        results = model(small_frame, verbose=False)
    except Exception as e:
        print(f"Detection error: {e}")
        return frame
    
    scale_x = width / RESIZE_WIDTH
    scale_y = height / (RESIZE_WIDTH * height / width)
    
    detections = []
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_name = model.names[int(box.cls[0])]
                    
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    detections.append((x1, y1, x2, y2, confidence, class_name))
                except Exception:
                    continue
    
    stable_detections = get_stable_detections(detections)
    
    for x1, y1, x2, y2, confidence, class_name in stable_detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    if stable_detections and current_time - last_speak_time > SPEAK_COOLDOWN:
        current_objects = [det[5] for det in stable_detections[:2]]
        
        if set(current_objects) != set(last_spoken_objects):
            if len(current_objects) == 1:
                speech_engine.speak(f"I see {current_objects[0]}")
            else:
                speech_engine.speak(f"I see {current_objects[0]} and {current_objects[1]}")
            
            last_spoken_objects = current_objects.copy()
            last_speak_time = current_time
    
    return frame

def process_distance_enhanced(frame, current_time):
    global last_speak_time
    
    height, width = frame.shape[:2]
    small_frame = cv2.resize(frame, (RESIZE_WIDTH, int(RESIZE_WIDTH * height / width)))
    
    try:
        results = model(small_frame, verbose=False)
    except Exception:
        return frame
    
    scale_x = width / RESIZE_WIDTH
    scale_y = height / (RESIZE_WIDTH * height / width)
    
    closest_distance = None
    closest_object = None
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_name = model.names[int(box.cls[0])]
                    
                    if confidence > CONFIDENCE_THRESHOLD:
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                        
                        box_width = x2 - x1
                        if box_width > 50:
                            distance = round((KNOWN_WIDTH * FOCAL_LENGTH) / box_width, 1)
                            
                            if distance < 50:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
                                cv2.putText(frame, f"{class_name}: {distance}m", (x1, y1-15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                                
                                if closest_distance is None or distance < closest_distance:
                                    closest_distance = distance
                                    closest_object = class_name
                except Exception:
                    continue
    
    if closest_object and current_time - last_speak_time > SPEAK_COOLDOWN:
        speech_engine.speak(f"{closest_object} at {closest_distance} meters")
        last_speak_time = current_time
    
    return frame

def process_directional_enhanced(frame, current_time):
    global last_speak_time, directional_object_index
    
    height, width = frame.shape[:2]
    small_frame = cv2.resize(frame, (RESIZE_WIDTH, int(RESIZE_WIDTH * height / width)))
    
    try:
        results = model(small_frame, verbose=False)
    except Exception:
        return frame
    
    scale_x = width / RESIZE_WIDTH
    scale_y = height / (RESIZE_WIDTH * height / width)
    
    # Draw zone divisions
    center_x = width // 2
    left_line = int(width * 0.35)
    right_line = int(width * 0.65)
    
    cv2.line(frame, (left_line, 0), (left_line, height), (100, 100, 100), 3)
    cv2.line(frame, (right_line, 0), (right_line, height), (100, 100, 100), 3)
    cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 255), 2)
    
    # Zone labels
    cv2.rectangle(frame, (20, 20), (120, 60), (0, 0, 0), -1)
    cv2.putText(frame, "LEFT", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    cv2.rectangle(frame, (center_x-60, 20), (center_x+60, 60), (0, 0, 0), -1)
    cv2.putText(frame, "CENTER", (center_x-50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.rectangle(frame, (width-120, 20), (width-20, 60), (0, 0, 0), -1)
    cv2.putText(frame, "RIGHT", (width-110, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    directional_objects = []
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_name = model.names[int(box.cls[0])]
                    
                    if confidence > CONFIDENCE_THRESHOLD:
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                        
                        box_area = (x2 - x1) * (y2 - y1)
                        if box_area > MIN_BOX_AREA:
                            x_center = (x1 + x2) // 2
                            y_center = (y1 + y2) // 2
                            
                            position = get_object_position_stable(x_center, width)
                            distance_cat = get_object_distance_category(y_center, height)
                            
                            obj_info = {
                                'label': class_name,
                                'position': position,
                                'distance_cat': distance_cat,
                                'coords': (x1, y1, x2, y2),
                                'center': (x_center, y_center),
                                'confidence': confidence
                            }
                            directional_objects.append(obj_info)
                except Exception:
                    continue
    
    directional_objects.sort(key=lambda obj: obj['center'][0])
    
    # Draw objects with continuous tracking
    for i, obj in enumerate(directional_objects):
        x1, y1, x2, y2 = obj['coords']
        label = obj['label']
        position = obj['position']
        distance_cat = obj['distance_cat']
        
        # Position-based colors
        if position == "left":
            color = (0, 255, 255)  # Yellow
        elif position == "right":
            color = (255, 0, 255)  # Magenta
        else:
            color = (0, 255, 0)    # Green
        
        thickness = 5 if i == directional_object_index % len(directional_objects) else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Directional arrows
        if position == "left":
            cv2.arrowedLine(frame, (x1 - 30, y1 + (y2-y1)//2), (x1 - 60, y1 + (y2-y1)//2), color, 4, tipLength=0.3)
        elif position == "right":
            cv2.arrowedLine(frame, (x2 + 30, y1 + (y2-y1)//2), (x2 + 60, y1 + (y2-y1)//2), color, 4, tipLength=0.3)
        else:
            cv2.arrowedLine(frame, ((x1+x2)//2, y1 - 30), ((x1+x2)//2, y1 - 60), color, 4, tipLength=0.3)
        
        # Labels
        label_text = f"{i+1}. {label} - {position.upper()}"
        if distance_cat == "close":
            label_text += " (CLOSE)"
        
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1-35), (x1 + text_size[0] + 10, y1-5), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (x1 + 5, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Directional announcements
    if directional_objects and current_time - last_speak_time > DIRECTIONAL_COOLDOWN:
        current_obj = directional_objects[directional_object_index % len(directional_objects)]
        
        announcement = f"{current_obj['label']} on your {current_obj['position']}"
        if current_obj['distance_cat'] == "close":
            announcement += " close to you"
        
        print(f"Directional announcement: {announcement}")
        speech_engine.speak_directional(announcement)
        
        directional_object_index += 1
        last_speak_time = current_time
        
        # Visual feedback
        cv2.rectangle(frame, (20, height - 90), (600, height - 30), (0, 0, 0), -1)
        cv2.putText(frame, f"ANNOUNCING: {announcement}", (30, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        progress_text = f"Object {directional_object_index % len(directional_objects) + 1} of {len(directional_objects)}"
        cv2.putText(frame, progress_text, (30, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    return frame

def process_text_enhanced(frame, current_time):
    global last_speak_time
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        text = pytesseract.image_to_string(gray, config='--psm 6').strip()
        
        if text and len(text) > 8:
            cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, 100), (0, 100, 0), 3)
            cv2.putText(frame, "TEXT DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            first_line = text.split('\n')[0][:50]
            cv2.putText(frame, first_line, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if current_time - last_speak_time > TEXT_COOLDOWN:
                speech_text = first_line[:60]
                speech_engine.speak(f"Text detected: {speech_text}")
                last_speak_time = current_time
        else:
            cv2.putText(frame, "Scanning for text...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
    
    except Exception as e:
        cv2.putText(frame, "Text detection error", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    return frame

def generate_frames():
    global cap
    
    if not initialize_camera():
        while True:
            error_frame = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Unavailable", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.2)
    
    while True:
        try:
            if not camera_on:
                pause_frame = np.zeros((600, 800, 3), dtype=np.uint8)
                cv2.putText(pause_frame, "PAUSED", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
                ret, buffer = cv2.imencode('.jpg', pause_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.2)
                continue
            
            success, frame = cap.read()
            if not success:
                continue
            
            processed_frame = process_frame_enhanced(frame)
            
            # Status overlay
            overlay = processed_frame.copy()
            cv2.rectangle(overlay, (5, 5), (350, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, processed_frame, 0.2, 0, processed_frame)
            
            cv2.putText(processed_frame, f"MODE: {mode.upper()}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if voice_commands_on:
                cv2.putText(processed_frame, "VOICE: ACTIVE", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(processed_frame, "Say 'enable voice'", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.putText(processed_frame, "SPEECH: ON" if speech_on else "SPEECH: OFF", (15, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if speech_on else (255, 0, 0), 2)
            
            cv2.putText(processed_frame, "ENHANCED STABLE MODE", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Frame generation error: {e}")
            time.sleep(0.1)

# Flask Routes
@app.route("/")
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Divya Drishti - Vision Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 30px;
        }
        .header {
            grid-column: 1 / -1;
            text-align: center;
            background: rgba(26, 26, 46, 0.9);
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 25px;
        }
        .header h1 {
            font-size: 2.8rem;
            color: #3498db;
            margin-bottom: 15px;
        }
        .video-section {
            background: rgba(26, 26, 46, 0.95);
            border-radius: 25px;
            padding: 25px;
            border: 3px solid rgba(52, 152, 219, 0.4);
        }
        .video-container {
            position: relative;
            width: 100%;
            border-radius: 20px;
            overflow: hidden;
        }
        #videoFeed {
            width: 100%;
            height: auto;
            border-radius: 20px;
        }
        .control-panel {
            background: rgba(26, 26, 46, 0.95);
            border-radius: 25px;
            padding: 30px;
            border: 3px solid rgba(52, 152, 219, 0.4);
        }
        .btn-grid {
            display: grid;
            gap: 12px;
            margin-bottom: 20px;
        }
        .btn-grid.two-col { grid-template-columns: 1fr 1fr; }
        .btn {
            padding: 15px 20px;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            color: white;
            transition: all 0.3s ease;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn-success { background: linear-gradient(135deg, #2ecc71, #27ae60); }
        .btn-danger { background: linear-gradient(135deg, #e74c3c, #c0392b); }
        .btn-primary { background: linear-gradient(135deg, #3498db, #2980b9); }
        .btn-warning { background: linear-gradient(135deg, #f39c12, #e67e22); }
        .btn-info { background: linear-gradient(135deg, #1abc9c, #16a085); }
        .btn.active {
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            box-shadow: 0 0 20px rgba(46, 204, 113, 0.4);
        }
        .section-title {
            font-size: 1.2rem;
            margin-bottom: 15px;
            color: #3498db;
            border-bottom: 2px solid rgba(52, 152, 219, 0.4);
            padding-bottom: 8px;
        }
        .status-display {
            background: rgba(22, 33, 62, 0.8);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        .status-item {
            text-align: center;
            padding: 10px;
            background: rgba(26, 26, 46, 0.9);
            border-radius: 10px;
        }
        .status-on { color: #2ecc71; }
        .status-off { color: #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Divya Drishti</h1>
            <p>Enhanced Anti-Jitter Directional Navigation System</p>
        </div>

        <div class="video-section">
            <div class="video-container">
                <img id="videoFeed" src="/video_feed" alt="Video Feed">
            </div>
        </div>

        <div class="control-panel">
            <div class="section-title">Voice Control</div>
            <div class="btn-grid two-col">
                <button class="btn btn-success" onclick="enableVoice()">Enable Voice</button>
                <button class="btn btn-danger" onclick="disableVoice()">Disable Voice</button>
            </div>

            <div class="section-title">Camera Control</div>
            <div class="btn-grid two-col">
                <button id="resumeBtn" class="btn btn-success active" onclick="resumeCamera()">Resume</button>
                <button id="pauseBtn" class="btn btn-danger" onclick="pauseCamera()">Pause</button>
            </div>

            <div class="section-title">Detection Modes</div>
            <div class="btn-grid">
                <button id="objectsBtn" class="btn btn-primary active" onclick="setMode('objects')">Objects</button>
                <button id="distanceBtn" class="btn btn-info" onclick="setMode('distance')">Distance</button>
                <button id="directionalBtn" class="btn btn-warning" onclick="setMode('directional')">Navigation</button>
                <button id="textBtn" class="btn btn-success" onclick="setMode('text')">Text Reading</button>
            </div>

            <div class="section-title">Speech Control</div>
            <div class="btn-grid">
                <button id="speechOnBtn" class="btn btn-success active" onclick="speechOn()">Speech On</button>
                <button id="speechOffBtn" class="btn btn-danger" onclick="speechOff()">Speech Off</button>
                <button class="btn btn-info" onclick="testSpeech()">Test Speech</button>
            </div>

            <div class="section-title">System Status</div>
            <div class="status-display">
                <div class="status-grid">
                    <div class="status-item">
                        <div>Camera</div>
                        <div id="cameraStatus" class="status-on">ON</div>
                    </div>
                    <div class="status-item">
                        <div>Mode</div>
                        <div id="modeStatus">OBJECTS</div>
                    </div>
                    <div class="status-item">
                        <div>Speech</div>
                        <div id="speechStatus" class="status-on">ON</div>
                    </div>
                    <div class="status-item">
                        <div>Voice</div>
                        <div id="voiceStatus" class="status-off">OFF</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function apiCall(endpoint) {
            try {
                const response = await fetch(endpoint, { method: 'POST' });
                const result = await response.json();
                console.log(`API ${endpoint}:`, result.status);
                return result;
            } catch (error) {
                console.error(`API error ${endpoint}:`, error);
                return { status: 'error' };
            }
        }

        async function enableVoice() { await apiCall('/enable_voice'); }
        async function disableVoice() { await apiCall('/disable_voice'); }
        async function resumeCamera() { 
            await apiCall('/resume');
            document.getElementById('resumeBtn').classList.add('active');
            document.getElementById('pauseBtn').classList.remove('active');
        }
        async function pauseCamera() { 
            await apiCall('/pause');
            document.getElementById('pauseBtn').classList.add('active');
            document.getElementById('resumeBtn').classList.remove('active');
        }

        async function setMode(mode) {
            await apiCall('/mode/' + mode);
            ['objects', 'distance', 'directional', 'text'].forEach(m => {
                document.getElementById(m + 'Btn').classList.toggle('active', m === mode);
            });
            document.getElementById('modeStatus').textContent = mode.toUpperCase();
        }

        async function speechOn() {
            await apiCall('/speech_on');
            document.getElementById('speechOnBtn').classList.add('active');
            document.getElementById('speechOffBtn').classList.remove('active');
            document.getElementById('speechStatus').textContent = 'ON';
            document.getElementById('speechStatus').className = 'status-on';
        }

        async function speechOff() {
            await apiCall('/speech_off');
            document.getElementById('speechOffBtn').classList.add('active');
            document.getElementById('speechOnBtn').classList.remove('active');
            document.getElementById('speechStatus').textContent = 'OFF';
            document.getElementById('speechStatus').className = 'status-off';
        }

        async function testSpeech() {
            await apiCall('/test_speech');
        }
    </script>
</body>
</html>"""

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/enable_voice", methods=['POST'])
def enable_voice():
    global voice_commands_on, voice_listener
    try:
        if voice_listener and not voice_listener.active:
            success = voice_listener.start()
            if success:
                voice_commands_on = True
                speech_engine.speak("Voice commands enabled", priority=True)
                return jsonify({"status": "success"})
        return jsonify({"status": "error"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/disable_voice", methods=['POST'])
def disable_voice():
    global voice_commands_on, voice_listener
    voice_commands_on = False
    if voice_listener and voice_listener.active:
        voice_listener.stop()
    speech_engine.speak("Voice commands disabled", priority=True)
    return jsonify({"status": "success"})

@app.route("/resume", methods=['POST'])
def resume():
    global camera_on
    camera_on = True
    speech_engine.speak("Camera resumed")
    return jsonify({"status": "success"})

@app.route("/pause", methods=['POST'])
def pause():
    global camera_on
    camera_on = False
    speech_engine.speak("Camera paused")
    return jsonify({"status": "success"})

@app.route("/mode/<mode_name>", methods=['POST'])
def set_mode(mode_name):
    global mode, directional_object_index
    valid_modes = ["objects", "distance", "directional", "text"]
    if mode_name in valid_modes:
        mode = mode_name
        if mode_name == "directional":
            directional_object_index = 0
            speech_engine.speak("Enhanced navigation mode activated")
        else:
            speech_engine.speak(f"{mode_name} mode activated")
        return jsonify({"status": "success"})
    return jsonify({"status": "error"})

@app.route("/speech_on", methods=['POST'])
def speech_on_route():
    global speech_on
    speech_on = True
    speech_engine.speak("Speech enabled")
    return jsonify({"status": "success"})

@app.route("/speech_off", methods=['POST'])
def speech_off_route():
    global speech_on
    speech_on = False
    return jsonify({"status": "success"})

@app.route("/test_speech", methods=['POST'])
def test_speech():
    speech_engine.speak("Enhanced directional navigation system active. Continuous object tracking enabled.")
    return jsonify({"status": "success"})

@app.route("/status")
def get_status():
    return jsonify({
        "camera": camera_on,
        "mode": mode,
        "speech": speech_on,
        "voice_commands": voice_commands_on,
        "voice_available": VOICE_AVAILABLE
    })

def cleanup():
    global cap, voice_listener
    try:
        if cap:
            cap.release()
        if voice_listener and voice_listener.active:
            voice_listener.stop()
        cv2.destroyAllWindows()
        print("Cleanup completed")
    except Exception as e:
        print(f"Cleanup error: {e}")

if __name__ == "__main__":
    import atexit
    import signal
    
    atexit.register(cleanup)
    
    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        cleanup()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("=" * 70)
        print("🎯 DIVYA DRISHTI - ENHANCED ANTI-JITTER VERSION")
        print("=" * 70)
        print("🔧 FEATURES:")
        print("   ✓ Continuous Object Tracking")
        print("   ✓ Anti-Jitter Position Detection")
        print("   ✓ Enhanced Left/Right/Center Detection")
        print("   ✓ Stable Directional Speech")
        print("   ✓ Voice Commands")
        print("   ✓ Multiple Detection Modes")
        print("=" * 70)
        print("📱 Access: http://localhost:5000")
        print("🎤 Voice: Say 'enable voice' to activate")
        print("🧭 Navigation: Enhanced directional mode")
        print("=" * 70)
        
        app.run(
            host="0.0.0.0",
            port=2000,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"Application error: {e}")
        cleanup()