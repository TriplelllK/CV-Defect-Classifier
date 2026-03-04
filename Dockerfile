FROM python:3.10-slim

# Системные пакеты
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY app ./app

# Порт приложения
EXPOSE 5000

# Старт сервера
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app.app:app"]
