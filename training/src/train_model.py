import warnings

warnings.filterwarnings(action="ignore")

from functools import partial
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from helper import load_config
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from omegaconf import DictConfig
from prefect import flow, get_run_logger, task
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


@task
def load_processed_data(path: DictConfig):
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        data[name] = pd.read_csv(path[name])
    return data


def get_objective(
    data: dict,
    config: DictConfig,
    space: dict,
):

    model = XGBClassifier(
        use_label_encoder=config.model.use_label_encoder,
        objective=config.model.objective,
        n_estimators=space["n_estimators"],
        max_depth=int(space["max_depth"]),
        gamma=space["gamma"],
        reg_alpha=int(space["reg_alpha"]),
        min_child_weight=int(space["min_child_weight"]),
        colsample_bytree=int(space["colsample_bytree"]),
    )

    evaluation = [
        (data["X_train"], data["y_train"]),
        (data["X_test"], data["y_test"]),
    ]

    model.fit(
        data["X_train"],
        data["y_train"],
        eval_set=evaluation,
        eval_metric=config.model.eval_metric,
        early_stopping_rounds=config.model.early_stopping_rounds,
    )
    prediction = model.predict(data["X_test"].values)
    accuracy = accuracy_score(data["y_test"], prediction)

    logger = get_run_logger()
    logger.info("SCORE:" + str(accuracy))
    return {"loss": -accuracy, "status": STATUS_OK, "model": model}


@task
def objective_fn(get_objective: Callable, data, config):
    return partial(get_objective, data, config)


@task
def get_space(config):
    return {
        "max_depth": hp.quniform("max_depth", **config.model.max_depth),
        "gamma": hp.uniform("gamma", **config.model.gamma),
        "reg_alpha": hp.quniform("reg_alpha", **config.model.reg_alpha),
        "reg_lambda": hp.uniform("reg_lambda", **config.model.reg_lambda),
        "colsample_bytree": hp.uniform(
            "colsample_bytree", **config.model.colsample_bytree
        ),
        "min_child_weight": hp.quniform(
            "min_child_weight", **config.model.min_child_weight
        ),
        "n_estimators": config.model.n_estimators,
        "seed": config.model.seed,
    }


@task
def optimize(objective: Callable, space: dict):
    trials = Trials()
    best_hyperparams = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
    )
    logger = get_run_logger()
    logger.info("The best hyperparameters are : " + "\n")
    logger.info(best_hyperparams)
    best_model = trials.results[
        np.argmin([r["loss"] for r in trials.results])
    ]["model"]
    return best_model


@task
def save_model(best_model, config):
    joblib.dump(best_model, config.model.path)


@flow
def train():
    """Function to train the model"""

    config = load_config()
    data = load_processed_data(config.processed)

    # Define space
    space = get_space(config)

    # Get objective
    objective = objective_fn(get_objective, data, config)

    # Find best model
    best_model = optimize(objective, space)

    # Save model
    save_model(best_model, config)


if __name__ == "__main__":
    train()
