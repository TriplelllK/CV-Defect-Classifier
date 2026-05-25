import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from app.model_utils import IMG_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app", "models", "neu_best_finetuned.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "app", "models", "class_names.txt")
EXAMPLE_DIR = os.path.join(BASE_DIR, "datasets", "validation", "images")


def find_sample_image(base_dir):
    if not os.path.isdir(base_dir):
        return None
    for cls in sorted(os.listdir(base_dir)):
        cls_path = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in sorted(os.listdir(cls_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                return os.path.join(cls_path, fname)
    return None


def find_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise RuntimeError("Backbone (вложенная модель) не найден")


def load_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0), arr.astype("uint8")


def make_gradcam(model, backbone, img_batch):
    preprocess = tf.keras.applications.resnet_v2.preprocess_input
    bb_idx = model.layers.index(backbone)
    head_layers = model.layers[bb_idx + 1:]

    with tf.GradientTape() as tape:
        conv_out = backbone(preprocess(img_batch), training=False)
        tape.watch(conv_out)
        x = conv_out
        for layer in head_layers:
            x = layer(x, training=False)
        preds = x
        class_channel = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), preds.numpy()


def show_gradcam(img_path, model, backbone, class_names, save_path=None):
    print(f"Изображение: {img_path}")
    img_batch, original = load_image(img_path)
    heatmap, preds = make_gradcam(model, backbone, img_batch)

    idx = int(np.argmax(preds[0]))
    pred_class = class_names[idx]
    confidence = float(preds[0][idx])

    h, w, _ = original.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = (255 * heatmap_resized).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = np.clip(heatmap_color * 0.4 + original, 0, 255).astype("uint8")

    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original); axes[0].set_title("Оригинал"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title(f"{pred_class} ({confidence*100:.2f}%)"); axes[2].axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Сохранено: {save_path}")
    plt.show()
    plt.close()
    print(f"Класс: {pred_class}, уверенность: {confidence*100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="путь к изображению")
    parser.add_argument("--save", help="сохранить результат в файл")
    args = parser.parse_args()

    img_path = args.image or find_sample_image(EXAMPLE_DIR)
    if not img_path or not os.path.isfile(img_path):
        raise SystemExit(f"Изображение не найдено: {img_path}")

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    backbone = find_backbone(model)

    show_gradcam(img_path, model, backbone, class_names, save_path=args.save)


if __name__ == "__main__":
    main()
