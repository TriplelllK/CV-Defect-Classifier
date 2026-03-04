import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

# Настройки


IMG_SIZE = (200, 200)
BATCH_SIZE = 16
EPOCHS = 30
SEED = 42


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Пути к данным
DATA_DIR_TRAIN = os.path.join(BASE_DIR, "datasets", "train", "images")
DATA_DIR_VAL   = os.path.join(BASE_DIR, "datasets", "validation", "images")

print(f"Base directory: {BASE_DIR}")
print(f"Train dir: {DATA_DIR_TRAIN}")
print(f"Val dir: {DATA_DIR_VAL}")

assert os.path.isdir(DATA_DIR_TRAIN), f"Train dir not found: {DATA_DIR_TRAIN}"
assert os.path.isdir(DATA_DIR_VAL), f"Val dir not found: {DATA_DIR_VAL}"

print("✓ Датасет найден!")


# Аугментация
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
)

train_data = train_datagen.flow_from_directory(
    directory=DATA_DIR_TRAIN,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED,
)

val_data = val_datagen.flow_from_directory(
    directory=DATA_DIR_VAL,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

num_classes = train_data.num_classes
class_indices = train_data.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(num_classes)]

print("Классы (по индексам):", class_names)


# Модель

base_model = tf.keras.applications.ResNet50V2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet',
)


base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = inputs

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.6)(x)

x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs, name="neu_resnet50v2")
model.summary()


# Папка для результатов
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# План изменения learning rate


initial_lr = 1e-4

def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        new_lr = lr * math.exp(-0.1)  
        return float(new_lr)


lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)

# Early stopping
earlystop_cb = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=7,
    restore_best_weights=True,
    mode="max",
    verbose=1,
)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=initial_lr,
        weight_decay=1e-5,
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


# Обучение


history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[lr_callback, earlystop_cb],
    verbose=1,
)


# Графики


def plot_history(history, save_path=None):
    # Сохраняет графики accuracy и loss
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="train_acc", marker='o')
    plt.plot(epochs, val_acc, label="val_acc", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Model Accuracy")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="train_loss", marker='o')
    plt.plot(epochs, val_loss, label="val_loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Model Loss")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранён: {save_path}")
    
    plt.show()

plot_history(history, save_path=os.path.join(RESULTS_DIR, "training_history.png"))


# Оценка


val_data.reset()
pred_probs = model.predict(val_data, verbose=1)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_data.classes
target_names = class_names

# Отчёт
report = classification_report(y_true, y_pred, target_names=target_names)
print("\nClassification report (VAL):")
print(report)

# Сохраняем отчёт в файл
report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("CLASSIFICATION REPORT (VALIDATION SET)\n")
    f.write("=" * 60 + "\n")
    f.write(report)
    f.write("\n\nТолкование метрик:\n")
    f.write("- Precision: доля верных предсказаний среди предсказанного класса\n")
    f.write("- Recall: доля верно предсказанных примеров класса\n")
    f.write("- F1-score: среднее гармоническое precision и recall\n")
    f.write("- Support: количество примеров класса в тестовом наборе\n")
print(f"✓ Classification report сохранён: {report_path}")

# Матрица ошибок
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 7))
plt.imshow(cm, cmap="Blues", aspect='auto')
plt.title("Confusion Matrix (Validation Set)", fontsize=14, fontweight='bold')
plt.colorbar(label='Count')

tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45, ha="right")
plt.yticks(tick_marks, target_names)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontweight='bold',
        )

plt.ylabel("True Label", fontweight='bold')
plt.xlabel("Predicted Label", fontweight='bold')
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix сохранена: {cm_path}")
plt.show()


# Сохранение модели
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "app", "models")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

final_model_path = os.path.join(MODEL_OUTPUT_DIR, "neu_best_finetuned.keras")
model.save(final_model_path)

# Сохранение имён классов
class_names_path = os.path.join(MODEL_OUTPUT_DIR, "class_names.txt")
with open(class_names_path, "w", encoding="utf-8") as f:
    for name in target_names:
        f.write(name + "\n")

print("\n" + "=" * 60)
print("✓ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
print("=" * 60)
print("\nСохранённые файлы для приложения:")
print(f"  📁 Модель: {final_model_path}")
print(f"  📄 Классы: {class_names_path}")
print("\n📊 Результаты метрик и графики:")
print(f"  📁 Папка: {RESULTS_DIR}/")
print("     📈 training_history.png - графики accuracy и loss")
print("     🔲 confusion_matrix.png - матрица ошибок")
print("     📋 classification_report.txt - детальный отчёт")
print("\nЧтобы запустить приложение:")
print("  python -m app.app")
print("  или: flask run")
print("=" * 60)
