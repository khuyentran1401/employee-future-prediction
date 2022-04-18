import warnings

warnings.filterwarnings(action="ignore")

import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def load_data(target_path: str, feature_path: str):
    y = pd.read_csv(target_path)
    X = pd.read_csv(feature_path)
    return y, X


def rename_columns(X: pd.DataFrame, map_columns: dict):
    return X.rename(columns=map_columns)


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame):
    return XGBClassifier(
        use_label_encoder=False, objective="binary:logistic"
    ).fit(X_train, y_train)


def predict(X_test: pd.DataFrame, model: XGBClassifier):
    return model.predict(X_test)


@hydra.main(config_path="../config", config_name="main")
def main(config: DictConfig):
    """Function to train the model"""

    target_path = abspath(config.processed.target.path)
    feature_path = abspath(config.processed.features.path)

    y, X = load_data(target_path, feature_path)
    X = rename_columns(X, config.map_columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )
    model = train_model(X_train, y_train)
    prediction = predict(X_test, model)
    f1 = f1_score(y_test, prediction)
    print(f"F1 Score of this model is {f1}.")


if __name__ == "__main__":
    main()
