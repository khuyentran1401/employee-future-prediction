import warnings

warnings.filterwarnings(action="ignore")

import joblib
import numpy as np
import pandas as pd
from helper import load_config
from omegaconf import DictConfig
from prefect import flow, get_run_logger, task
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


@task
def load_data(path: DictConfig):
    X_test = pd.read_csv(path.X_test)
    y_test = pd.read_csv(path.y_test)
    return X_test, y_test


@task
def load_model(model_path: str):
    return joblib.load(model_path)


@task
def predict(model: XGBClassifier, X_test: pd.DataFrame):
    return model.predict(X_test)


@task
def get_metrics(y_test: np.array, prediction: np.array):
    logger = get_run_logger()

    f1 = f1_score(y_test, prediction)
    logger.info(f"F1 Score of this model is {f1}.")

    accuracy = accuracy_score(y_test, prediction)
    logger.info(f"Accuracy Score of this model is {accuracy}.")


@flow
def evaluate():

    config = load_config()

    # Load data and model
    X_test, y_test = load_data(config.processed)

    model = load_model(config.model.path)

    # Get predictions
    prediction = predict(model, X_test)

    # Get metrics
    get_metrics(y_test, prediction)


if __name__ == "__main__":
    evaluate()
