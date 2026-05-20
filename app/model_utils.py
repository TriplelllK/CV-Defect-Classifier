import os
import numpy as np
from PIL import Image
import tensorflow as tf

preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

IMG_SIZE = (200, 200)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "neu_best_finetuned.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.txt")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]


def preprocess_image(file) -> np.ndarray:
    file.stream.seek(0)
    img = Image.open(file).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype="float32")
    return preprocess_input(np.expand_dims(arr, axis=0))


def predict_defect(file):
    preds = model.predict(preprocess_image(file), verbose=0)[0]
    idx = int(np.argmax(preds))
    probs = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    return class_names[idx], float(preds[idx]), probs
