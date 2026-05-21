import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from app.model_utils import IMG_SIZE

keras = tf.keras
layers = tf.keras.layers


BATCH_SIZE = 16
EPOCHS = 30
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_TRAIN = os.path.join(BASE_DIR, "datasets", "train", "images")
DATA_DIR_VAL = os.path.join(BASE_DIR, "datasets", "validation", "images")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "app", "models")

for d in (DATA_DIR_TRAIN, DATA_DIR_VAL):
    if not os.path.isdir(d):
        raise SystemExit(f"Не найдена папка с данными: {d}")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR_TRAIN,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED,
)
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR_VAL,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Классы:", class_names)

preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
], name="augment")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


base_model = keras.applications.ResNet50V2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="neu_resnet50v2")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True, mode="max", verbose=1),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)


def plot_history(h, save_path):
    acc, val_acc = h.history["accuracy"], h.history["val_accuracy"]
    loss, val_loss = h.history["loss"], h.history["val_loss"]
    ep = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ep, acc, label="train_acc")
    plt.plot(ep, val_acc, label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(ep, loss, label="train_loss")
    plt.plot(ep, val_loss, label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


plot_history(history, os.path.join(RESULTS_DIR, "training_history.png"))


y_true, y_pred = [], []
for x_batch, y_batch in val_ds:
    probs = model.predict(x_batch, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1))
    y_true.extend(np.argmax(y_batch.numpy(), axis=1))

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
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()


model.save(os.path.join(MODEL_OUTPUT_DIR, "neu_best_finetuned.keras"))
with open(os.path.join(MODEL_OUTPUT_DIR, "class_names.txt"), "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print("\nГотово. Модель и отчёты сохранены.")
