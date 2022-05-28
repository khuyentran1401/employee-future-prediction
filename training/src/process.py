import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from patsy import dmatrices
from sklearn.model_selection import train_test_split


def get_data(raw_path: str):
    data = pd.read_csv(raw_path)
    return data


def get_features(target: str, features: list, data: pd.DataFrame):
    feature_str = " + ".join(features)
    y, X = dmatrices(
        f"{target} ~ {feature_str} - 1", data=data, return_type="dataframe"
    )
    return y, X


def rename_columns(X: pd.DataFrame):
    X.columns = X.columns.str.replace("[", "_", regex=True).str.replace(
        "]", "", regex=True
    )
    return X


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def process_data(config: DictConfig):
    """Function to process the data"""

    data = get_data(abspath(config.raw.path))

    y, X = get_features(config.process.target, config.process.features, data)

    X = rename_columns(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    # Save data
    X_train.to_csv(abspath(config.processed.X_train.path), index=False)
    X_test.to_csv(abspath(config.processed.X_test.path), index=False)
    y_train.to_csv(abspath(config.processed.y_train.path), index=False)
    y_test.to_csv(abspath(config.processed.y_test.path), index=False)


if __name__ == "__main__":
    process_data()
