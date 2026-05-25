import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from app.model_utils import IMG_SIZE


BATCH_SIZE = 16
EPOCHS_HEAD = 15
EPOCHS_FT = 15
SEED = 42
FINE_TUNE_LAYERS = 20

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_TRAIN = os.path.join(BASE_DIR, "datasets", "train", "images")
DATA_DIR_VAL = os.path.join(BASE_DIR, "datasets", "validation", "images")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "app", "models")


def make_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_TRAIN, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=True, seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_VAL, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=False,
    )
    class_names = train_ds.class_names

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ], name="augment")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y),
                            num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names


def build_model(num_classes):
    preprocess = tf.keras.applications.resnet_v2.preprocess_input
    base = tf.keras.applications.ResNet50V2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = preprocess(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="neu_resnet50v2"), base


def callbacks():
    return [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True, verbose=1),
    ]


def plot_history(histories, save_path):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]
    ep = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ep, acc, label="train_acc")
    plt.plot(ep, val_acc, label="val_acc")
    plt.axvline(x=len(histories[0].history["accuracy"]), color="gray",
                linestyle="--", label="fine-tune start")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(ep, loss, label="train_loss")
    plt.plot(ep, val_loss, label="val_loss")
    plt.axvline(x=len(histories[0].history["loss"]), color="gray",
                linestyle="--", label="fine-tune start")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate(model, val_ds, class_names):
    val_probs = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(val_probs, axis=1)
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in val_ds])

    report = str(classification_report(y_true, y_pred, target_names=class_names))
    print("\nClassification report (VAL):")
    print(report)

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("CLASSIFICATION REPORT (VALIDATION)\n")
        f.write("=" * 60 + "\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix (VAL)")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    thr = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thr else "black")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    for d in (DATA_DIR_TRAIN, DATA_DIR_VAL):
        if not os.path.isdir(d):
            raise SystemExit(f"Не найдена папка с данными: {d}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    train_ds, val_ds, class_names = make_datasets()
    print("Классы:", class_names)

    model, base = build_model(len(class_names))
    model.summary()

    # Этап 1: учим только голову
    print("\n--- Этап 1: обучение головы ---")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_HEAD, callbacks=callbacks(), verbose=1)

    # Этап 2: fine-tuning топ-20 слоёв backbone
    print(f"\n--- Этап 2: fine-tuning последних {FINE_TUNE_LAYERS} слоёв ---")
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_FT, callbacks=callbacks(), verbose=1)

    plot_history([h1, h2], os.path.join(RESULTS_DIR, "training_history.png"))
    evaluate(model, val_ds, class_names)

    model.save(os.path.join(MODEL_OUTPUT_DIR, "neu_best_finetuned.keras"))
    with open(os.path.join(MODEL_OUTPUT_DIR, "class_names.txt"), "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    print("\nГотово. Модель и отчёты сохранены.")


if __name__ == "__main__":
    main()
