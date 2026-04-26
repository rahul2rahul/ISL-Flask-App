"""
app.py  —  ISL Greetings Flask App  (Render free-tier, 512 MB safe)
====================================================================
MEMORY STRATEGY:
  mBERT  → HuggingFace Serverless Inference API (no local RAM used)
  Sign   → hf_hub_download .keras → load locally (~20-50 MB, safe)

KERAS VERSION FIX:
  Model was trained with Keras 2 (bundled in tensorflow 2.x).
  Render installs tensorflow-cpu==2.15.0 + tf-keras==2.15.0.
  We must set TF_USE_LEGACY_KERAS=1 AND import from tf_keras
  (not keras) so TimeDistributed shape handling matches training.
"""

import os, random, base64, cv2, json, time
import numpy as np
import requests

# ── MUST be set before ANY tensorflow / keras import ─────────────
os.environ["USE_TF"]                            = "1"
os.environ["USE_TORCH"]                         = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]              = "2"
os.environ["TF_USE_LEGACY_KERAS"]               = "1"   # force Keras 2 inside TF 2.15

import tensorflow as tf

# Use tf_keras (the standalone Keras 2 package) so load_model
# uses exactly the same Keras version the model was saved with.
try:
    import tf_keras as keras_compat
    _load_model = keras_compat.models.load_model
    _preprocess = keras_compat.applications.mobilenet_v2.preprocess_input
    print("Using tf_keras (Keras 2 compat) for load_model")
except ImportError:
    # Fallback — should not happen if requirements.txt is correct
    from tensorflow import keras as keras_compat
    _load_model = keras_compat.models.load_model
    _preprocess = keras_compat.applications.mobilenet_v2.preprocess_input
    print("Fallback: using tensorflow.keras for load_model")

from flask import Flask, render_template, request, jsonify, url_for
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════
# 1.  CONFIG
# ═══════════════════════════════════════════════════════════════════
HF_REPO_ID = "rahul2025/isl"
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{HF_REPO_ID}"

SIGN_LOCAL = "/tmp/isl_sign_model.keras"
LMAP_LOCAL = "/tmp/isl_label_map.json"

# ═══════════════════════════════════════════════════════════════════
# 2.  LABEL MAPS
# ═══════════════════════════════════════════════════════════════════
LABEL2ID = {
    "HELLO": 0, "GOOD_MORNING": 1, "GOOD_AFTERNOON": 2,
    "GOOD_EVENING": 3, "GOOD_NIGHT": 4, "HOW_ARE_YOU": 5,
    "ALRIGHT": 6, "PLEASED": 7, "THANK_YOU": 8,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

LABEL_TO_BENGALI = {
    "HELLO":          "হ্যালো",
    "GOOD_MORNING":   "শুভ সকাল",
    "GOOD_AFTERNOON": "শুভ অপরাহ্ন",
    "GOOD_EVENING":   "শুভ সন্ধ্যা",
    "GOOD_NIGHT":     "শুভ রাত্রি",
    "HOW_ARE_YOU":    "আপনি কেমন আছেন",
    "ALRIGHT":        "সব ঠিক আছে",
    "PLEASED":        "খুশি",
    "THANK_YOU":      "ধন্যবাদ",
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
    "THANK_YOU":      "Thank You",
}

# ── Keyword fallback (used if HF API is loading / rate-limited) ──
_KEYWORDS = {
    "hello": "HELLO", "hi": "HELLO", "hey": "HELLO",
    "নমস্কার": "HELLO", "হ্যালো": "HELLO", "হাই": "HELLO",
    "good morning": "GOOD_MORNING", "morning": "GOOD_MORNING",
    "শুভ সকাল": "GOOD_MORNING", "সকাল": "GOOD_MORNING",
    "good afternoon": "GOOD_AFTERNOON", "afternoon": "GOOD_AFTERNOON",
    "শুভ অপরাহ্ন": "GOOD_AFTERNOON",
    "good evening": "GOOD_EVENING", "evening": "GOOD_EVENING",
    "শুভ সন্ধ্যা": "GOOD_EVENING", "সন্ধ্যা": "GOOD_EVENING",
    "good night": "GOOD_NIGHT", "night": "GOOD_NIGHT",
    "শুভ রাত্রি": "GOOD_NIGHT", "রাত": "GOOD_NIGHT",
    "how are you": "HOW_ARE_YOU", "how do you do": "HOW_ARE_YOU",
    "আপনি কেমন": "HOW_ARE_YOU", "কেমন আছেন": "HOW_ARE_YOU",
    "alright": "ALRIGHT", "fine": "ALRIGHT", "okay": "ALRIGHT",
    "ok": "ALRIGHT", "i'm fine": "ALRIGHT",
    "সব ঠিক": "ALRIGHT", "ভালো আছি": "ALRIGHT",
    "pleased": "PLEASED", "glad to meet": "PLEASED",
    "nice to meet": "PLEASED", "meet you": "PLEASED",
    "খুশি": "PLEASED",
    "thank you": "THANK_YOU", "thanks": "THANK_YOU",
    "thank": "THANK_YOU", "ধন্যবাদ": "THANK_YOU",
}

def _keyword_fallback(text: str):
    t = text.lower().strip()
    for phrase, label in sorted(_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if phrase in t:
            return label, 0.75
    return "HELLO", 0.50

# ═══════════════════════════════════════════════════════════════════
# 3.  DOWNLOAD SIGN MODEL FILES ONLY  (mBERT never downloaded)
# ═══════════════════════════════════════════════════════════════════
def _download_sign_model():
    import shutil
    print("=== Checking sign model ===")

    if not os.path.isfile(SIGN_LOCAL):
        print("  Downloading model_cnn_bilstm.keras …")
        tmp = hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = "model/model_cnn_bilstm.keras",
            token     = HF_TOKEN or None,
            local_dir = "/tmp/hf_dl",
        )
        shutil.copy(tmp, SIGN_LOCAL)
        print(f"  Saved → {SIGN_LOCAL}")
    else:
        print(f"  Already exists: {SIGN_LOCAL}")

    if not os.path.isfile(LMAP_LOCAL):
        print("  Downloading label_map.json …")
        tmp = hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = "model/label_map.json",
            token     = HF_TOKEN or None,
            local_dir = "/tmp/hf_dl",
        )
        shutil.copy(tmp, LMAP_LOCAL)
        print(f"  Saved → {LMAP_LOCAL}")
    else:
        print(f"  Already exists: {LMAP_LOCAL}")

    print("=== Sign model ready ===")


_download_sign_model()

# ═══════════════════════════════════════════════════════════════════
# 4.  LOAD SIGN MODEL with tf_keras (Keras 2 — matches training)
# ═══════════════════════════════════════════════════════════════════
SEQ_LEN_SIGN = 15
IMG_SIZE     = 96

print("Loading CNN+BiLSTM sign model …")
sign_model = _load_model(SIGN_LOCAL)
print("Sign model loaded ✓")

with open(LMAP_LOCAL, encoding="utf-8") as f:
    label_map = json.load(f)

# ═══════════════════════════════════════════════════════════════════
# 5.  INTENT PREDICTION via HF Serverless Inference API
# ═══════════════════════════════════════════════════════════════════
def _normalise_label(raw: str) -> str:
    r = raw.upper().strip()
    if r in LABEL2ID:
        return r
    if r.startswith("LABEL_"):
        try:
            return ID2LABEL[int(r.split("_", 1)[1])]
        except (ValueError, KeyError):
            pass
    r2 = r.replace(" ", "_")
    if r2 in LABEL2ID:
        return r2
    for k in LABEL2ID:
        if k in r or r in k:
            return k
    return "HELLO"


def predict_intent(text: str, retries: int = 3):
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    for attempt in range(retries):
        try:
            resp = requests.post(
                HF_INFERENCE_URL,
                headers = headers,
                json    = {"inputs": text},
                timeout = 30,
            )

            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data and isinstance(data[0], list):
                    data = data[0]
                if isinstance(data, list) and data:
                    best  = max(data, key=lambda x: x.get("score", 0))
                    label = _normalise_label(best.get("label", "HELLO"))
                    return label, round(best.get("score", 0.9), 4)

            elif resp.status_code == 503:
                try:
                    wait = float(resp.json().get("estimated_time", 20))
                except Exception:
                    wait = 20.0
                wait = min(wait, 25.0)
                print(f"  HF API: model loading, sleeping {wait:.0f}s "
                      f"(attempt {attempt + 1}/{retries}) …")
                time.sleep(wait)
                continue

            else:
                print(f"  HF API error {resp.status_code}: {resp.text[:300]}")
                break

        except requests.exceptions.Timeout:
            print(f"  HF API timeout (attempt {attempt + 1}/{retries})")
        except Exception as exc:
            print(f"  HF API exception: {exc}")
            break

    print("  Using keyword fallback for intent detection.")
    return _keyword_fallback(text)

# ═══════════════════════════════════════════════════════════════════
# 6.  SIGN PREDICTION HELPERS
# ═══════════════════════════════════════════════════════════════════
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
    indices = np.linspace(0, n - 1, SEQ_LEN_SIGN, dtype=int)
    clip    = []
    for i in indices:
        rgb     = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        clip.append(resized.astype("float32"))
    clip = np.array(clip)
    clip = _preprocess(clip)
    return clip[np.newaxis, ...]


def get_video_for_label(label):
    videos       = []
    video_folder = os.path.join(app.static_folder, "videos", label)
    if os.path.exists(video_folder):
        files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
        if files:
            selected = random.choice(files)
            videos.append(
                url_for("static", filename=f"videos/{label}/{selected}")
            )
    return videos

# ═══════════════════════════════════════════════════════════════════
# 7.  ROUTES
# ═══════════════════════════════════════════════════════════════════
@app.route("/")
def home():
    return render_template("home.html")


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
        "videos":          get_video_for_label(intent_label),
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

    clip            = frames_to_clip(frames)
    preds           = sign_model.predict(clip, verbose=0)
    class_idx       = int(np.argmax(preds[0]))
    confidence      = float(np.max(preds[0]))
    predicted_label = label_map.get(str(class_idx), "UNKNOWN")

    return jsonify({
        "predicted_label": predicted_label,
        "display_name":    LABEL_TO_DISPLAY.get(predicted_label, predicted_label),
        "bengali":         LABEL_TO_BENGALI.get(predicted_label, ""),
        "confidence":      round(confidence, 4),
        "videos":          get_video_for_label(predicted_label),
    })


# ═══════════════════════════════════════════════════════════════════
# 8.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
