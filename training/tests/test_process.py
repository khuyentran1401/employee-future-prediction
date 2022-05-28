import pandas as pd
from pandera import Check, Column, DataFrameSchema
from pytest_steps import test_steps

from training.src.process import get_features, rename_columns


@test_steps("get_features_step", "rename_columns_step")
def test_processs_suite(test_step, steps_data):
    if test_step == "get_features_step":
        get_features_step(steps_data)
    elif test_step == "rename_columns_step":
        rename_columns_step(steps_data)


def get_features_step(steps_data):
    data = pd.DataFrame(
        {
            "Education": ["Bachelors", "Masters"],
            "City": ["Bangalore", "Prune"],
            "PaymentTier": [2, 3],
            "Age": [30, 21],
            "Gender": ["Male", "Female"],
            "EverBenched": ["No", "Yes"],
            "ExperienceInCurrentDomain": [2, 3],
            "LeaveOrNot": [0, 1],
        }
    )
    features = [
        "City",
        "PaymentTier",
        "Age",
        "Gender",
        "EverBenched",
        "ExperienceInCurrentDomain",
    ]
    target = "LeaveOrNot"
    y, X = get_features(target, features, data)
    schema = DataFrameSchema(
        {
            "City[Bangalore]": Column(float, Check.isin([0.0, 1.0])),
            "City[Prune]": Column(float, Check.isin([0.0, 1.0])),
            "Gender[T.Male]": Column(float, Check.isin([0.0, 1.0])),
            "EverBenched[T.Yes]": Column(float, Check.isin([0.0, 1.0])),
            "PaymentTier": Column(float, Check.isin([1, 2, 3])),
            "Age": Column(float, Check.greater_than(10)),
            "ExperienceInCurrentDomain": Column(
                float, Check.greater_than_or_equal_to(0)
            ),
        }
    )
    schema.validate(X)
    steps_data.X = X


def rename_columns_step(steps_data):
    processed_X = rename_columns(steps_data.X)
    assert list(processed_X.columns) == [
        "City_Bangalore",
        "City_Prune",
        "Gender_T.Male",
        "EverBenched_T.Yes",
        "PaymentTier",
        "Age",
        "ExperienceInCurrentDomain",
    ]
