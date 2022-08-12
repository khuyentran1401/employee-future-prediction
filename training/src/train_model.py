import warnings

warnings.filterwarnings(action="ignore")

from datetime import timedelta

import bentoml
import joblib
import numpy as np
import pandas as pd
from helper import load_config
from omegaconf import DictConfig
from prefect import flow, get_run_logger, task
from prefect.tasks import task_input_hash
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV


@task
def load_processed_data(path: DictConfig):
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        data[name] = pd.read_csv(path[name])
    return data


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def train_model(data: dict, config: DictConfig):
    model = KNeighborsClassifier()
    params = tuple(config.n_neighbors)
    grid = BayesSearchCV(
        model,
        search_spaces={"n_neighbors": params},
        cv=config.cv,
        scoring=config.scoring,
        random_state=config.random_state,
    )
    res = grid.fit(data["X_train"], data["y_train"])
    return res.best_estimator_


@task
def save_model(best_model, config):
    joblib.dump(best_model, config.model.path)
    bentoml.sklearn.save_model(config.model.name, best_model)


@flow
def train():
    """Function to train the model"""

    config = load_config()
    data = load_processed_data(config.processed)

    # Train model
    best_model = train_model(data, config.model)

    # Save model
    save_model(best_model, config)


if __name__ == "__main__":
    train()
