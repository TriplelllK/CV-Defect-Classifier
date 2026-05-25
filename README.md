# CV Defect Classifier

Учебный проект: классификация дефектов поверхности металла на 6 классов по фото.
Модель — ResNet50V2 (transfer learning) на датасете NEU Surface Defects.
Веб-интерфейс и REST API сделаны на Flask.

## Стек

- Python 3.10+
- TensorFlow / Keras (ResNet50V2)
- Flask + Gunicorn
- NumPy, Pillow
- opencv-python (только для Grad-CAM)
- scikit-learn (метрики), matplotlib (графики)

## Классы дефектов

`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`

Источник датасета: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

## Структура

```
app/                 — Flask приложение
  app.py             — роуты (/, /api/predict, /api/health)
  model_utils.py     — препроцессинг и предсказание
  models/            — обученная модель и class_names.txt
  templates/, static/
training/
  train_neu_model.py — обучение модели
  grad_cam_demo.py   — визуализация Grad-CAM
datasets/            — train/validation изображения
images/              — графики и примеры для README
Dockerfile, requirements.txt
```

## Установка

```bash
git clone https://github.com/TriplelllK/CV-Defect-Classifier.git
cd CV-Defect-Classifier

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/Mac

pip install -r requirements.txt
```

## Обучение

```bash
python -m training.train_neu_model
```

Обучение идёт в два этапа: сначала только «голова» поверх замороженного ResNet50V2, потом fine-tuning последних 20 слоёв с маленьким lr. Модель сохраняется в `app/models/neu_best_finetuned.keras`, графики и classification_report — в `training/results/`.

На CPU занимает ~15 минут.

## Запуск приложения

```bash
python -m app.app
```

Открыть http://localhost:5000, загрузить картинку — получить класс и вероятности.

## REST API

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict
```

Ответ:

```json
{
  "success": true,
  "prediction": {
    "class": "scratches",
    "confidence": 0.9982,
    "probabilities": {
      "crazing": 0.0,
      "inclusion": 0.0,
      "patches": 0.0015,
      "pitted_surface": 0.0,
      "rolled-in_scale": 0.0003,
      "scratches": 0.9982
    }
  }
}
```

Health-check:

```bash
curl http://localhost:5000/api/health
# {"status":"ok","model_loaded":true}
```

## Docker

```bash
docker build -t cv-defect-classifier .
docker run -p 5000:5000 cv-defect-classifier
```

## Модель

ResNet50V2 (imagenet weights) + голова:
GlobalAveragePooling → Dense(512, relu) + BN + Dropout(0.4) → Dense(256, relu) + BN + Dropout(0.4) → Dense(6, softmax).

Препроцессинг — `tf.keras.applications.resnet_v2.preprocess_input`.
Loss: categorical_crossentropy.
Callbacks: ReduceLROnPlateau(val_loss), EarlyStopping(val_loss).

Двухэтапное обучение:
1. Backbone заморожен, голова учится с Adam(lr=1e-3).
2. Размораживаем последние 20 слоёв backbone, дообучаем с Adam(lr=1e-5).

## Метрики

Validation делится 50/50 на val (для EarlyStopping) и test (для финальной оценки). На test после fine-tuning получается **~98% accuracy** при F1 0.96–1.00 по классам.

![training history](images/training_history.png)
![confusion matrix](images/confusion_matrix.png)

## Grad-CAM

```bash
python -m training.grad_cam_demo
# конкретное изображение
python -m training.grad_cam_demo --image datasets/validation/images/scratches/scratches_290.jpg
# сохранить результат в файл
python -m training.grad_cam_demo --save out.png
```

Показывает, на какие области картинки модель «смотрит» при предсказании.

![grad-cam](images/grad_cam4.png)
