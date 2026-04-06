from flask import Flask, render_template, request, jsonify, url_for, redirect, session
import os, random, base64, cv2, json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'supersecretkey' 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USERNAME = "admin"
PASSWORD = "1234"

MODEL_PATH = "./final_model"
tokenizer     = AutoTokenizer.from_pretrained(MODEL_PATH)
intent_model  = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
intent_model.to(DEVICE)
intent_model.eval()

SIGN_MODEL_PATH = "./model/model_cnn_bilstm.keras"
LABEL_MAP_PATH  = "./model/label_map.json"

sign_model = load_model(SIGN_MODEL_PATH)
SEQ_LEN    = 15
IMG_SIZE   = 96

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)  


LABEL_TO_BENGALI = {
    "HELLO":          "হ্যালো",
    "GOOD_MORNING":   "শুভ সকাল",
    "GOOD_AFTERNOON": "শুভ অপরাহ্ন",
    "GOOD_EVENING":   "শুভ সন্ধ্যা",
    "GOOD_NIGHT":     "শুভ রাত্রি",
    "HOW_ARE_YOU":    "আপনি কেমন আছেন",
    "ALRIGHT":        "সব ঠিক আছে",
    "PLEASED":        "খুশি",
    "THANK_YOU":      "ধন্যবাদ"
}

LABEL_TO_DISPLAY = {
    "HELLO":          "Hello",
    "GOOD_MORNING":   "Good Morning",
    "GOOD_AFTERNOON": "Good Afternoon",
    "GOOD_EVENING":   "Good Evening",
    "GOOD_NIGHT":     "Good Night",
    "HOW_ARE_YOU":    "How Are You?",
    "ALRIGHT":        "Alright / I'm Fine",
    "PLEASED":        "Pleased to Meet You",
    "THANK_YOU":      "Thank You"
}


def predict_intent(text: str):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding=True, max_length=32
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = intent_model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)
        idx     = torch.argmax(probs, dim=1).item()
        conf    = probs[0][idx].item()
    label = intent_model.config.id2label[idx]
    return label, round(conf, 4)



@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            return redirect("/home")
        else:
            return render_template("login.html", error="Invalid username or password")
    return render_template("login.html", error=None)


def decode_frames(frame_b64_list):
    frames = []
    for b64 in frame_b64_list:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        arr       = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    return frames

def frames_to_clip(frames):
    n       = len(frames)
    indices = np.linspace(0, n - 1, SEQ_LEN, dtype=int)
    clip    = []
    for i in indices:
        frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        resized   = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
        clip.append(resized.astype("float32"))
    clip = np.array(clip)
    clip = preprocess_input(clip)   # [-1, 1] — matches training
    return clip[np.newaxis, ...]

def get_video_for_label(label):
    videos       = []
    video_folder = os.path.join(app.static_folder, "videos", label)
    if os.path.exists(video_folder):
        files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
        if files:
            selected = random.choice(files)
            videos.append(url_for("static", filename=f"videos/{label}/{selected}"))
    return videos



@app.route("/home")
def home():
    if not session.get('logged_in'):
        return redirect("/")
    return render_template("home.html")  # your existing main page


@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    return redirect("/")

@app.route("/process_speech", methods=["POST"])
def process_speech():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No input text"}), 400
    intent_label, confidence = predict_intent(text)
    return jsonify({
        "input_text":      text,
        "predicted_label": intent_label,
        "display_name":    LABEL_TO_DISPLAY.get(intent_label, intent_label),
        "bengali":         LABEL_TO_BENGALI.get(intent_label, ""),
        "confidence":      confidence,
        "videos":          get_video_for_label(intent_label)
    })

@app.route("/predict_sign", methods=["POST"])
def predict_sign():
    data       = request.get_json()
    frame_list = data.get("frames", [])
    if not frame_list:
        return jsonify({"error": "No frames received"}), 400
    frames = decode_frames(frame_list)
    if len(frames) < 5:
        return jsonify({"error": "Too few valid frames captured"}), 400
    clip  = frames_to_clip(frames)
    preds = sign_model.predict(clip, verbose=0)
    class_idx       = int(np.argmax(preds[0]))
    confidence      = float(np.max(preds[0]))
    predicted_label = label_map.get(str(class_idx), "UNKNOWN")
    return jsonify({
        "predicted_label": predicted_label,
        "display_name":    LABEL_TO_DISPLAY.get(predicted_label, predicted_label),
        "bengali":         LABEL_TO_BENGALI.get(predicted_label, ""),
        "confidence":      round(confidence, 4),
        "videos":          get_video_for_label(predicted_label)
    })


if __name__ == "__main__":
    app.run(debug=True)