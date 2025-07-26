FROM python:3.10-slim

# Системні залежності (якщо будуть потрібні)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Робоча директорія в контейнері
WORKDIR /app

# Копіюємо файли проєкту
COPY . .

# Встановлюємо Python-залежності
RUN pip install --upgrade pip && pip install -r requirements.txt

# Запускаємо тренування
CMD ["python", "train.py"]
