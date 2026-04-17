import os
import gc
import sys
import random, base64, cv2, json, logging
import numpy as np
import tensorflow as tf
import requests
import time

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, render_template, request, jsonify, url_for
from huggingface_hub import hf_hub_download

# ── ENV ─────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# Hugging Face API (NO MODEL LOADING LOCALLY)
API_URL = "https://api-inference.huggingface.co/models/rahul2025/isl"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# ── logging ─────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── LABEL MAPS ─────────────────────────────────────
LABEL_TO_BENGALI = {
    "HELLO": "হ্যালো",
    "GOOD_MORNING": "শুভ সকাল",
    "GOOD_AFTERNOON": "শুভ অপরাহ্ন",
    "GOOD_EVENING": "শুভ সন্ধ্যা",
    "GOOD_NIGHT": "শুভ রাত্রি",
    "HOW_ARE_YOU": "আপনি কেমন আছেন",
    "ALRIGHT": "সব ঠিক আছে",
    "PLEASED": "খুশি",
    "THANK_YOU": "ধন্যবাদ",
}

LABEL_TO_DISPLAY = {
    "HELLO": "Hello",
    "GOOD_MORNING": "Good Morning",
    "GOOD_AFTERNOON": "Good Afternoon",
    "GOOD_EVENING": "Good Evening",
    "GOOD_NIGHT": "Good Night",
    "HOW_ARE_YOU": "How Are You?",
    "ALRIGHT": "Alright / I'm Fine",
    "PLEASED": "Pleased to Meet You",
    "THANK_YOU": "Thank You",
}

# ── SIGN MODEL ─────────────────────────────────────
_sign_model = None
_label_map = None

SEQ_LEN_SIGN = 15
IMG_SIZE = 96
HF_SIGN_ID = "rahul2025/isl-sign"


def get_sign_model():
    global _sign_model, _label_map

    if _sign_model is None:
        log.info("Downloading sign model...")
        model_path = hf_hub_download(
            repo_id=HF_SIGN_ID,
            filename="model_cnn_bilstm.keras",
            token=HF_TOKEN,
        )
        _sign_model = load_model(model_path, compile=False)
        gc.collect()

    if _label_map is None:
        label_map_path = hf_hub_download(
            repo_id=HF_SIGN_ID,
            filename="label_map.json",
            token=HF_TOKEN,
        )
        with open(label_map_path) as f:
            _label_map = json.load(f)

    return _sign_model, _label_map


# ── TEXT PREDICTION (API) ──────────────────────────
def predict_intent(text):
    for _ in range(3):
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        result = response.json()

        if isinstance(result, dict) and "error" in result:
            if "loading" in result["error"]:
                time.sleep(3)
                continue
            return "ERROR", 0.0

        return result[0][0]["label"], round(result[0][0]["score"], 4)

    return "ERROR", 0.0


# ── FRAME HELPERS ──────────────────────────────────
def decode_frames(frame_b64_list):
    frames = []
    for b64 in frame_b64_list:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            arr = np.frombuffer(base64.b64decode(b64), np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        except:
            pass
    return frames


def frames_to_clip(frames):
    indices = np.linspace(0, len(frames) - 1, SEQ_LEN_SIGN, dtype=int)
    clip = []
    for i in indices:
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        clip.append(resized.astype("float32"))
    clip = preprocess_input(np.array(clip))
    return clip[np.newaxis, ...]


def get_video_for_label(label):
    folder = os.path.join(app.static_folder, "videos", label)
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
        if files:
            return [url_for("static", filename=f"videos/{label}/{random.choice(files)}")]
    return []


# ── ROUTES ─────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/process_speech", methods=["POST"])
def process_speech():
    data = request.get_json() or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No input text"}), 400

    label, conf = predict_intent(text)

    return jsonify({
        "input_text": text,
        "predicted_label": label,
        "display_name": LABEL_TO_DISPLAY.get(label, label),
        "bengali": LABEL_TO_BENGALI.get(label, ""),
        "confidence": conf,
        "videos": get_video_for_label(label),
    })


@app.route("/predict_sign", methods=["POST"])
def predict_sign():
    data = request.get_json() or {}
    frames = decode_frames(data.get("frames", []))

    if len(frames) < 5:
        return jsonify({"error": "Too few frames"}), 400

    model, label_map = get_sign_model()

    clip = frames_to_clip(frames)
    preds = model.predict(clip, verbose=0)

    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))
    label = label_map.get(str(idx), "UNKNOWN")

    return jsonify({
        "predicted_label": label,
        "display_name": LABEL_TO_DISPLAY.get(label, label),
        "bengali": LABEL_TO_BENGALI.get(label, ""),
        "confidence": round(conf, 4),
        "videos": get_video_for_label(label),
    })


# ── ENTRY ──────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
