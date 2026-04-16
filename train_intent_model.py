import os
import random
import json

# ── MUST be set before any other import ─────────────────────────
os.environ["USE_TF"]                            = "1"
os.environ["USE_TORCH"]                         = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]              = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from transformers import (
    BertTokenizerFast,
    TFBertForSequenceClassification,
    create_optimizer,
)

# ═══════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
MODEL_NAME   = "bert-base-multilingual-cased"
DATA_PATH    = "./data/train.csv"
OUTPUT_DIR   = "./final_model"
NUM_LABELS   = 9
MAX_LENGTH   = 32
BATCH_SIZE   = 16
EPOCHS       = 20
LR           = 3e-5
WARMUP_RATIO = 0.1
AUG_FACTOR   = 6
LABEL_SMOOTH = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL2ID = {
    "HELLO": 0, "GOOD_MORNING": 1, "GOOD_AFTERNOON": 2,
    "GOOD_EVENING": 3, "GOOD_NIGHT": 4, "HOW_ARE_YOU": 5,
    "ALRIGHT": 6, "PLEASED": 7, "THANK_YOU": 8,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Save label maps for app.py
with open(os.path.join(OUTPUT_DIR, "id2label.json"), "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in ID2LABEL.items()}, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUTPUT_DIR, "label2id.json"), "w", encoding="utf-8") as f:
    json.dump(LABEL2ID, f, ensure_ascii=False, indent=2)

# ═══════════════════════════════════════════════════════════════════
# 2.  TEXT AUGMENTATION  (pure Python — zero external libs)
# ═══════════════════════════════════════════════════════════════════
_SYNONYMS = {
    # English
    "hello":     ["hi", "hey", "greetings", "howdy", "hiya"],
    "hi":        ["hello", "hey", "greetings", "howdy"],
    "hey":       ["hello", "hi", "yo", "howdy"],
    "morning":   ["dawn", "daybreak", "early hours"],
    "good":      ["great", "wonderful", "fine", "nice", "lovely"],
    "afternoon": ["midday", "noon", "daytime"],
    "evening":   ["dusk", "twilight", "sundown"],
    "night":     ["nighttime", "overnight", "late evening"],
    "fine":      ["okay", "alright", "well", "great"],
    "okay":      ["fine", "alright", "ok", "good"],
    "alright":   ["okay", "fine", "good", "all right"],
    "pleased":   ["glad", "happy", "delighted", "thrilled"],
    "thank":     ["thanks", "grateful", "appreciate"],
    "thanks":    ["thank you", "many thanks", "cheers"],
    "meet":      ["see", "greet", "encounter"],
    "doing":     ["feeling", "faring", "holding up"],
    "sleep":     ["rest", "slumber", "nap"],
    "sweet":     ["pleasant", "lovely", "peaceful"],
    "how":       ["in what way", "tell me how"],
    "have":      ["get", "enjoy"],
    # Bengali
    "নমস্কার":   ["হ্যালো", "হাই", "প্রণাম"],
    "ভালো":      ["চমৎকার", "সুন্দর", "উত্তম"],
    "ধন্যবাদ":   ["কৃতজ্ঞ", "শুকরিয়া", "আপনাকে ধন্যবাদ"],
    "শুভ":       ["মঙ্গল", "কল্যাণ", "সুন্দর"],
    "সকাল":     ["ভোর", "প্রভাত"],
    "সন্ধ্যা":   ["গোধূলি", "সন্ধ্যাবেলা"],
    "রাত":       ["রাতে", "নিশি"],
    "আছেন":     ["আছ", "আছো", "রয়েছেন"],
    "কেমন":     ["কীভাবে", "কিরকম"],
    "আপনি":     ["তুমি", "তুই"],
    "খুশি":     ["আনন্দিত", "প্রসন্ন", "সুখী"],
}

def _replace(w):
    key = w.lower().strip(".,!?।")
    return random.choice(_SYNONYMS[key]) if key in _SYNONYMS else w

def synonym_replace(text, p=0.30):
    return " ".join(_replace(w) if random.random() < p else w for w in text.split())

def random_delete(text, p=0.12):
    words = text.split()
    if len(words) <= 2:
        return text
    kept = [w for w in words if random.random() > p]
    return " ".join(kept) if kept else text

def random_swap(text, n=1):
    words = text.split()
    if len(words) < 2:
        return text
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)

def insert_word(text):
    words = text.split()
    if not words:
        return text
    syn = random.choice(list(_SYNONYMS.values()))
    words.insert(random.randint(0, len(words)), random.choice(syn))
    return " ".join(words)

_OPS = [synonym_replace, random_delete, random_swap, insert_word]

def augment(text, label, n=AUG_FACTOR):
    out = []
    for _ in range(n):
        aug = random.choice(_OPS)(text)
        if aug.strip() and aug != text:
            out.append({"text": aug, "label": label})
    return out

# ═══════════════════════════════════════════════════════════════════
# 3.  LOAD & AUGMENT DATA
# ═══════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)
df["label_id"] = df["label"].map(LABEL2ID)

aug_rows = []
for _, row in df.iterrows():
    aug_rows.extend(augment(row["text"], row["label"]))

aug_df             = pd.DataFrame(aug_rows)
aug_df["label_id"] = aug_df["label"].map(LABEL2ID)

full_df = pd.concat([df, aug_df], ignore_index=True).sample(frac=1, random_state=42)
print(f"Total samples after augmentation: {len(full_df)}")
print(full_df["label"].value_counts().to_string())

train_df, val_df = train_test_split(
    full_df, test_size=0.15, random_state=42,
    stratify=full_df["label_id"]
)
print(f"\nTrain: {len(train_df)}  |  Val: {len(val_df)}")

# ═══════════════════════════════════════════════════════════════════
# 4.  TOKENISE  (return_tensors="tf" — pure TF tensors, no torch)
# ═══════════════════════════════════════════════════════════════════
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenise(texts):
    return tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="tf",
    )

train_enc = tokenise(train_df["text"])
val_enc   = tokenise(val_df["text"])

train_labels = tf.constant(train_df["label_id"].values, dtype=tf.int32)
val_labels   = tf.constant(val_df["label_id"].values,   dtype=tf.int32)

def make_dataset(enc, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((
        {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        },
        labels,
    ))
    if shuffle:
        ds = ds.shuffle(2048)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_enc, train_labels, shuffle=True)
val_ds   = make_dataset(val_enc,   val_labels)

# ═══════════════════════════════════════════════════════════════════
# 5.  MODEL
#
#     With torch UNINSTALLED:
#       from_pt=True → HuggingFace sees no torch → falls back to
#       downloading the native tf_model.h5 weights automatically.
#
#     With torch INSTALLED (even broken):
#       from_pt=True → tries "import torch" → crashes.
#       Solution: uninstall torch completely (see top of file).
# ═══════════════════════════════════════════════════════════════════
print("\nLoading mBERT TF weights (downloading ~700 MB on first run)...")
print("This is fine — torch is not needed. Weights come as tf_model.h5.")

model = TFBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels = NUM_LABELS,
    id2label   = ID2LABEL,
    label2id   = LABEL2ID,
    from_pt    = True,   # safe only when torch is NOT installed
)

print("mBERT loaded successfully in TF mode.")

# ── Cosine LR schedule with warm-up ─────────────────────────────
num_train_steps  = len(train_ds) * EPOCHS
num_warmup_steps = int(num_train_steps * WARMUP_RATIO)

optimizer, _ = create_optimizer(
    init_lr           = LR,
    num_train_steps   = num_train_steps,
    num_warmup_steps  = num_warmup_steps,
    weight_decay_rate = 0.01,
)

# ── Label-smoothing loss ─────────────────────────────────────────
def smooth_loss(y_true, logits):
    y_true   = tf.cast(tf.squeeze(y_true), tf.int32)
    num_cls  = tf.shape(logits)[-1]
    y_oh     = tf.one_hot(y_true, num_cls)
    y_smooth = (y_oh * (1.0 - LABEL_SMOOTH)
                + LABEL_SMOOTH / tf.cast(num_cls, tf.float32))
    log_prob = tf.nn.log_softmax(logits, axis=-1)
    return -tf.reduce_mean(tf.reduce_sum(y_smooth * log_prob, axis=-1))

model.compile(
    optimizer = optimizer,
    loss      = smooth_loss,
    metrics   = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# ═══════════════════════════════════════════════════════════════════
# 6.  CALLBACKS
# ═══════════════════════════════════════════════════════════════════
CKPT_PATH = os.path.join(OUTPUT_DIR, "best_weights")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath          = CKPT_PATH,
        monitor           = "val_accuracy",
        save_best_only    = True,
        save_weights_only = True,
        mode              = "max",
        verbose           = 1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor               = "val_accuracy",
        patience              = 5,
        restore_best_weights  = True,
        verbose               = 1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = 3,
        min_lr   = 1e-7,
        verbose  = 1,
    ),
]

# ═══════════════════════════════════════════════════════════════════
# 7.  TRAIN
# ═══════════════════════════════════════════════════════════════════
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs          = EPOCHS,
    callbacks       = callbacks,
)

# ═══════════════════════════════════════════════════════════════════
# 8.  EVALUATE
# ═══════════════════════════════════════════════════════════════════
all_logits = model.predict(val_ds).logits
preds      = np.argmax(all_logits, axis=1)
true       = val_df["label_id"].values
names      = [ID2LABEL[i] for i in range(NUM_LABELS)]

print("\n=== Classification Report ===")
print(classification_report(true, preds, target_names=names, zero_division=0))

p, r, f1, sup = precision_recall_fscore_support(
    true, preds, average=None, zero_division=0
)
pd.DataFrame({
    "label": names, "precision": p,
    "recall": r, "f1_score": f1, "support": sup,
}).to_csv(os.path.join(OUTPUT_DIR, "per_class_metrics.csv"), index=False)

mp, mr, mf1, _ = precision_recall_fscore_support(
    true, preds, average="macro", zero_division=0
)
wp, wr, wf1, _ = precision_recall_fscore_support(
    true, preds, average="weighted", zero_division=0
)
pd.DataFrame({
    "average_type": ["macro", "weighted"],
    "precision":    [mp, wp],
    "recall":       [mr, wr],
    "f1_score":     [mf1, wf1],
}).to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"), index=False)

cm = confusion_matrix(true, preds)
pd.DataFrame(cm, index=names, columns=names).to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
)
print("All metrics saved.")

# ═══════════════════════════════════════════════════════════════════
# 9.  SAVE as TF SavedModel  (app.py loads with from_pretrained)
#     Produces:
#       final_model/config.json
#       final_model/tf_model.h5    ← native TF weights, no torch needed
#       final_model/vocab.txt      ← tokenizer files
#       final_model/tokenizer_config.json
#       final_model/special_tokens_map.json
# ═══════════════════════════════════════════════════════════════════
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved    → {OUTPUT_DIR}/tf_model.h5")
print(f"Tokenizer saved → {OUTPUT_DIR}/")
print("\napp.py loads this with TFBertForSequenceClassification.from_pretrained()")
print("Torch is NOT needed to load tf_model.h5 — it is native TF format.")