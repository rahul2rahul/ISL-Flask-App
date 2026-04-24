"""
app.py  —  ISL Greetings Flask App  (Render-ready, HuggingFace model download)
================================================================================
Models are stored in HuggingFace repo  rahul2025/isl
and are downloaded ONCE at startup into /tmp (always writable on Render free tier).

Dependencies (requirements.txt):
    flask
    tensorflow-cpu
    transformers==4.40.2
    tokenizers>=0.15,<0.20
    opencv-python-headless
    huggingface_hub
    numpy
    requests

Zero torch.  Runs on Render free tier (512 MB RAM).
"""

import os, random, base64, cv2, json
import numpy as np

# ── CRITICAL: set BEFORE any transformers import ─────────────────
os.environ["USE_TF"]                            = "1"
os.environ["USE_TORCH"]                         = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]              = "2"

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from flask import Flask, render_template, request, jsonify, url_for

from transformers import BertTokenizerFast, TFBertForSequenceClassification
from huggingface_hub import hf_hub_download, snapshot_download

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════
# 1.  LABEL MAPS
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

# ═══════════════════════════════════════════════════════════════════
# 2.  HUGGINGFACE REPO CONFIG
# ═══════════════════════════════════════════════════════════════════
HF_REPO_ID   = "rahul2025/isl"
HF_TOKEN     = os.environ.get("HF_TOKEN")          # set in Render dashboard

# Local cache paths (writable on Render free tier)
INTENT_LOCAL = "/tmp/isl_intent_model"
SIGN_LOCAL   = "/tmp/isl_sign_model.keras"
LMAP_LOCAL   = "/tmp/isl_label_map.json"

# ═══════════════════════════════════════════════════════════════════
# 3.  DOWNLOAD MODELS AT STARTUP
# ═══════════════════════════════════════════════════════════════════
def download_models():
    """
    Download model files from HuggingFace into /tmp.
    Skips files that already exist (container restarts are rare on Render
    but this saves time during local dev restarts).
    """
    print("=== Checking / downloading models from HuggingFace ===")

    # ── 3a. Intent model (whole folder) ──────────────────────────
    # snapshot_download fetches all files in final_model/ subfolder.
    # They land at  /tmp/isl_intent_model/  preserving sub-paths.
    if not os.path.isdir(INTENT_LOCAL) or not os.listdir(INTENT_LOCAL):
        print("  Downloading intent model (mBERT)…")
        os.makedirs(INTENT_LOCAL, exist_ok=True)
        snapshot_download(
            repo_id        = HF_REPO_ID,
            token          = HF_TOKEN,
            allow_patterns = ["final_model/*"],
            local_dir      = "/tmp/hf_cache_isl",
        )
        # snapshot_download puts files under  /tmp/hf_cache_isl/final_model/
        # Move them up one level so INTENT_LOCAL is the model root.
        import shutil
        src = "/tmp/hf_cache_isl/final_model"
        if os.path.isdir(src):
            shutil.copytree(src, INTENT_LOCAL, dirs_exist_ok=True)
        print(f"  Intent model ready at {INTENT_LOCAL}")
    else:
        print(f"  Intent model already at {INTENT_LOCAL}, skipping download.")

    # ── 3b. Sign model (.keras) ───────────────────────────────────
    if not os.path.isfile(SIGN_LOCAL):
        print("  Downloading sign model (CNN+BiLSTM)…")
        path = hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = "model/model_cnn_bilstm.keras",
            token     = HF_TOKEN,
            local_dir = "/tmp",
        )
        # hf_hub_download returns the actual path; rename to our constant
        os.makedirs(os.path.dirname(SIGN_LOCAL) or ".", exist_ok=True)
        if path != SIGN_LOCAL:
            import shutil
            shutil.copy(path, SIGN_LOCAL)
        print(f"  Sign model ready at {SIGN_LOCAL}")
    else:
        print(f"  Sign model already at {SIGN_LOCAL}, skipping download.")

    # ── 3c. Label map ─────────────────────────────────────────────
    if not os.path.isfile(LMAP_LOCAL):
        print("  Downloading label_map.json…")
        path = hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = "model/label_map.json",
            token     = HF_TOKEN,
            local_dir = "/tmp",
        )
        if path != LMAP_LOCAL:
            import shutil
            shutil.copy(path, LMAP_LOCAL)
        print(f"  Label map ready at {LMAP_LOCAL}")
    else:
        print(f"  Label map already at {LMAP_LOCAL}, skipping download.")

    print("=== All models ready ===")


download_models()

# ═══════════════════════════════════════════════════════════════════
# 4.  LOAD mBERT INTENT MODEL  (TF — no torch)
# ═══════════════════════════════════════════════════════════════════
print("Loading mBERT tokenizer and classifier…")
tokenizer    = BertTokenizerFast.from_pretrained(INTENT_LOCAL)
intent_model = TFBertForSequenceClassification.from_pretrained(
    INTENT_LOCAL,
    num_labels = len(LABEL2ID),
    id2label   = ID2LABEL,
    label2id   = LABEL2ID,
)
print("mBERT loaded.")

# ═══════════════════════════════════════════════════════════════════
# 5.  LOAD SIGN MODEL  (CNN+BiLSTM Keras)
# ═══════════════════════════════════════════════════════════════════
SEQ_LEN_SIGN = 15
IMG_SIZE     = 96

print("Loading CNN+BiLSTM sign model…")
sign_model = load_model(SIGN_LOCAL)
print("Sign model loaded.")

with open(LMAP_LOCAL, encoding="utf-8") as f:
    label_map = json.load(f)   # {"0": "ALRIGHT", "1": "GOOD_MORNING", ...}

# ═══════════════════════════════════════════════════════════════════
# 6.  INTENT PREDICTION  — mBERT, pure TF, zero torch
# ═══════════════════════════════════════════════════════════════════
def predict_intent(text: str):
    encoded = tokenizer(
        text,
        return_tensors = "tf",
        truncation     = True,
        padding        = True,
        max_length     = 32,
    )
    outputs = intent_model(**encoded, training=False)
    probs   = tf.nn.softmax(outputs.logits, axis=-1)
    idx     = int(tf.argmax(probs, axis=-1).numpy()[0])
    conf    = float(probs.numpy()[0][idx])
    label   = ID2LABEL.get(idx, "UNKNOWN")
    return label, round(conf, 4)

# ═══════════════════════════════════════════════════════════════════
# 7.  SIGN PREDICTION HELPERS
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
    clip = preprocess_input(clip)
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

# ═══════════════════════════════════════════════════════════════════
# 8.  ROUTES
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
# 9.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
