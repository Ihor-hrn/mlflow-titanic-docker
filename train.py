import pandas as pd
import boto3
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_csv_from_s3(bucket, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))

def train_model():
    mlflow.set_experiment("Titanic Experiment")

    with mlflow.start_run():
        # 1. Читання з S3
        df = load_csv_from_s3("mlflow-titanic-ihorhrynch", "datasets/titanic.csv")

        # 2. Мінімальна обробка
        df = df.dropna(subset=['Age'])
        X = df[['Pclass', 'Age', 'SibSp', 'Parch']]
        y = df['Survived']

        # 3. Тренування
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        # 4. Логування в MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"✅ Модель натреновано. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model()
