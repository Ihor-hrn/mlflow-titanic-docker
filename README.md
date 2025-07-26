# MLflow Titanic in Docker 🐳🎓

Цей проєкт — приклад виконання домашнього завдання з контейнеризації моделі машинного навчання з використанням:

- AWS S3 для зберігання датасету
- MLflow для логування експериментів
- Docker для ізоляції тренувального середовища

---

## 📁 Структура проєкту

mlflow-titanic-docker/
├── train.py # Скрипт тренування моделі
├── requirements.txt # Залежності
├── Dockerfile # Опис контейнера
├── .dockerignore # Ігноровані файли для Docker


---

## 🚀 Запуск проєкту

### 1. Побудова образу

```bash
docker build -t titanic-trainer .
2. Запуск з передачі AWS-ключів

docker run --rm ^
 -e AWS_ACCESS_KEY_ID=YOUR_KEY ^
 -e AWS_SECRET_ACCESS_KEY=YOUR_SECRET ^
 titanic-trainer
Скрипт читає датасет titanic.csv з вашого бакета S3, тренує модель RandomForestClassifier і логує її в MLflow.

🪣 Приклад бакета S3
markdown
Копіювати
Редагувати
mlflow-titanic-ihorhrynch
└── datasets/
    └── titanic.csv
📈 MLflow
Логування:

параметри (n_estimators)

метрики (accuracy)

модель (sklearn)

UI доступний локально через mlflow ui
