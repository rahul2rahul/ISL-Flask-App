import os
import gc

# ── CRITICAL: set BEFORE any transformers import ─────────────────
os.environ["USE_TF"]                            = "1"
os.environ["USE_TORCH"]                         = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ── TF memory optimizations ───────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]             = "3"
os.environ["CUDA_VISIBLE_DEVICES"]             = ""        # CPU only
os.environ["TF_ENABLE_ONEDNN_OPTS"]            = "0"

import random, base64, cv2, json
import numpy as np
from huggingface_hub import hf_hub_download
import tensorflow as tf

# ── Limit TF to use only what it needs ───────────────────────────
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, render_template, request, jsonify, url_for
from transformers import BertTokenizerFast, TFBertForSequenceClassification

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
# 2.  LAZY-LOAD GLOBALS  (load once on first request, not at startup)
# ═══════════════════════════════════════════════════════════════════
_tokenizer    = None
_intent_model = None
_sign_model   = None
_label_map    = None

SEQ_LEN_SIGN = 15
IMG_SIZE     = 96
HF_MODEL_ID  = "rahul2025/isl"


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BertTokenizerFast.from_pretrained(HF_MODEL_ID)
    return _tokenizer


def get_intent_model():
    global _intent_model
    if _intent_model is None:
        # ── Download the SavedModel / TF weights folder ───────────
        # from_pt=True needs torch; instead expect the repo to have
        # a saved_model/ subfolder OR tf_model.h5.
        # If your HF repo only has pytorch_model.bin, you MUST
        # convert once offline and push tf_model.h5 / saved_model.
        #
        # Here we attempt TF loading; if the repo has no TF weights
        # this will raise a clear error at first request (not crash
        # the dyno at boot).
        _intent_model = TFBertForSequenceClassification.from_pretrained(
            HF_MODEL_ID,
            from_pt    = False,   # set True only if you have torch installed
            num_labels = len(LABEL2ID),
            id2label   = ID2LABEL,
            label2id   = LABEL2ID,
        )
        gc.collect()
    return _intent_model


def get_sign_model():
    global _sign_model, _label_map
    if _sign_model is None:
        model_path = hf_hub_download(
            repo_id  = "rahul2025/isl-sign",
            filename = "model_cnn_bilstm.keras",
        )
        _sign_model = load_model(model_path, compile=False)   # compile=False saves ~30 MB
        gc.collect()

    if _label_map is None:
        label_map_path = hf_hub_download(
            repo_id  = "rahul2025/isl-sign",
            filename = "label_map.json",
        )
        with open(label_map_path, encoding="utf-8") as f:
            _label_map = json.load(f)

    return _sign_model, _label_map


# ═══════════════════════════════════════════════════════════════════
# 3.  INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════════════
def predict_intent(text: str):
    tokenizer    = get_tokenizer()
    intent_model = get_intent_model()

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
# 4.  ROUTES
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

    sign_model, label_map = get_sign_model()

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
# 5.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
