import pandas as pd
from helper import load_config
from patsy import dmatrices
from prefect import flow, task
from sklearn.model_selection import train_test_split


@task
def get_data(raw_path: str):
    data = pd.read_csv(raw_path)
    return data


@task
def get_features(target: str, features: list, data: pd.DataFrame):
    feature_str = " + ".join(features)
    y, X = dmatrices(
        f"{target} ~ {feature_str} - 1", data=data, return_type="dataframe"
    )
    return y, X


@task
def rename_columns(X: pd.DataFrame):
    X.columns = X.columns.str.replace("[", "_", regex=True).str.replace(
        "]", "", regex=True
    )
    return X


@task
def split_train_test(X: pd.DataFrame, y: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@task
def save_data(data: dict, config):
    for name, df in data.items():
        df.to_csv(config.processed[name], index=False)


@flow
def process_data():
    """Function to process the data"""

    config = load_config()
    data = get_data(config.raw.path)

    y, X = get_features(config.process.target, config.process.features, data)

    X = rename_columns(X)
    data = split_train_test(X, y)
    save_data(data, config)


if __name__ == "__main__":
    process_data()
