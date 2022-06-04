import joblib
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import ModelErrorAnalysis
from hydra import compose, initialize
from hydra.utils import to_absolute_path as abspath

from training.src.train_model import load_data


def test_xgboost():

    with initialize(version_base=None, config_path="../../config"):
        config = compose(config_name="main")

    model_path = abspath(config.model.path)
    model = joblib.load(model_path)
    X_train, X_test, y_train, y_test = load_data(config.processed)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_ds = Dataset(train_df, label="LeaveOrNot")
    validation_ds = Dataset(test_df, label="LeaveOrNot")

    check = ModelErrorAnalysis(min_error_model_score=0.3)
    check.run(train_ds, validation_ds, model)
