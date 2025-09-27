import cv2
import pytesseract
import numpy as np
import threading
import os
import base64
import pyttsx3
from flask import Flask, render_template, Response, jsonify, request
from spellchecker import SpellChecker
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser

app = Flask(__name__)

# ---------------- CONFIG ----------------
TESSERACT_LANG = "eng"
CUSTOM_CONFIG = "--oem 3 --psm 6"
latest_ocr_text = ""
ocr_lock = threading.Lock()

# SpellChecker
spell = SpellChecker()

# ---------------- TTS ENGINE ----------------
engine = pyttsx3.init()

def speak_text(text):
    """Speak OCR text asynchronously"""
    def run():
        engine.say(text)
        engine.runAndWait()
    t = threading.Thread(target=run)
    t.start()

# ---------------- WHOOSH INDEX ----------------
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")
    schema = Schema(id=ID(stored=True, unique=True), content=TEXT(stored=True))
    ix = create_in("indexdir", schema)
else:
    ix = open_dir("indexdir")

def add_to_index(doc_id, content):
    """Save OCR text into Whoosh index"""
    writer = ix.writer()
    writer.update_document(id=str(doc_id), content=content)
    writer.commit()

def search_index(query_str):
    """Search OCR text in Whoosh index"""
    results_list = []
    with ix.searcher() as searcher:
        parser = QueryParser("content", ix.schema)
        query = parser.parse(query_str)
        results = searcher.search(query, limit=10)
        for r in results:
            results_list.append({"id": r["id"], "content": r["content"]})
    return results_list

# ---------------- CAMERA (OBS VirtualCam) ----------------
OBS_CAM_INDEX = 2  

if os.name == "nt":  # Windows
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
else:  # macOS/Linux
    cap = cv2.VideoCapture(2)

if not cap.isOpened():
    raise RuntimeError(f"❌ OBS Virtual Camera not available at index {OBS_CAM_INDEX}")

# ---------------- UTILITIES ----------------
def preprocess_image(image):
    """Convert frame to a binarized version for OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text(image):
    """OCR + smart spell correction"""
    data = pytesseract.image_to_string(
        image, lang=TESSERACT_LANG, config=CUSTOM_CONFIG
    ).strip()

    words = data.split()
    corrected_words = []
    for w in words:
        if w.lower() not in spell:
            corrected_words.append(spell.correction(w) or w)
        else:
            corrected_words.append(w)
    return " ".join(corrected_words)

# ---------------- STREAM GENERATOR ----------------
def gen_raw_frames():
    """Stream raw camera frames"""
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/raw_feed')
def raw_feed():
    return Response(gen_raw_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_ocr')
def capture_ocr():
    global latest_ocr_text
    success, frame = cap.read()
    if not success:
        return jsonify({"error": "Failed to capture frame"})

    processed = preprocess_image(frame)
    text = extract_text(processed)

    with ocr_lock:
        latest_ocr_text = text

    # Add to search index
    if text.strip():
        add_to_index(doc_id=hash(text), content=text)
        speak_text(text)  # 🔊 Speak OCR result

    # Encode processed image as base64
    _, buffer = cv2.imencode('.jpg', processed)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"text": text, "image": img_base64})

@app.route('/search')
def search():
    q = request.args.get("q", "")
    if not q:
        return jsonify({"results": []})
    results = search_index(q)
    return jsonify({"results": results})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("✅ Flask running at http://127.0.0.1:5000")
    app.run(debug=True, threaded=True)
