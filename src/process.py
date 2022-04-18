import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from patsy import dmatrices


def get_data(raw_path: str):
    data = pd.read_csv(raw_path)
    return data


def get_features_without_intercept(
    target: str, features: list, data: pd.DataFrame
):
    feature_str = " + ".join(features)
    y, X = dmatrices(
        f"{target} ~ {feature_str} - 1", data=data, return_type="dataframe"
    )
    return y, X


def get_features_with_intercept(
    target: str, features: list, data: pd.DataFrame
):
    feature_str = " + ".join(features)
    y, X = dmatrices(
        f"{target} ~ {feature_str}", data=data, return_type="dataframe"
    )
    return y, X


@hydra.main(config_path="../config", config_name="main")
def process_data(config: DictConfig):
    """Function to process the data"""

    data = get_data(abspath(config.raw.path))

    if config.process.has_intercept:
        y, X = get_features_with_intercept(
            config.process.target, config.process.features, data
        )
    else:
        y, X = get_features_without_intercept(
            config.process.target, config.process.features, data
        )

    y.to_csv(abspath(config.processed.target.path), index=False)
    X.to_csv(abspath(config.processed.features.path), index=False)


if __name__ == "__main__":
    process_data()
