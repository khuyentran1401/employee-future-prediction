import joblib
import pandas as pd
import streamlit as st
from hydra import compose, initialize
from hydra.utils import to_absolute_path as abspath
from patsy import dmatrix

with initialize(version_base=None, config_path="../../config"):
    config = compose(config_name="main")
    FEATURES = config.process.features
    MODEL_PATH = abspath(config.model.path)


def add_dummy_data(df: pd.DataFrame):
    """Add dummy rows so that patsy can create features similar to the train dataset"""
    rows = {
        "City": ["Bangalore", "New Delhi", "Pune"],
        "Gender": ["Male", "Female", "Female"],
        "EverBenched": ["Yes", "Yes", "No"],
        "PaymentTier": [0, 0, 0],
        "Age": [0, 0, 0],
        "ExperienceInCurrentDomain": [0, 0, 0],
    }
    dummy_df = pd.DataFrame(rows)
    return pd.concat([df, dummy_df])


def rename_columns(X: pd.DataFrame):
    X.columns = X.columns.str.replace("[", "_", regex=True).str.replace(
        "]", "", regex=True
    )
    return X


def transform_data(df: pd.DataFrame):
    """Transform the data"""
    dummy_df = add_dummy_data(df)
    feature_str = " + ".join(FEATURES)
    dummy_X = dmatrix(f"{feature_str} - 1", dummy_df, return_type="dataframe")
    dummy_X = rename_columns(dummy_X)
    return dummy_X.iloc[0, :].values.reshape(1, -1)


def get_inputs():
    """Get inputs from users on streamlit"""
    st.title("Predict employee future")

    data = {}

    data["City"] = st.selectbox(
        "City Office Where Posted",
        options=["Bangalore", "Pune", "New Delhi"],
    )
    data["PaymentTier"] = st.selectbox(
        "Payment Tier",
        options=[1, 2, 3],
        help="payment tier: 1: highest 2: mid level 3:lowest",
    )
    data["Age"] = st.number_input(
        "Current Age", min_value=15, step=1, value=20
    )
    data["Gender"] = st.selectbox("Gender", options=["Male", "Female"])
    data["EverBenched"] = st.selectbox(
        "Ever Kept Out of Projects for 1 Month or More", options=["No", "Yes"]
    )
    data["ExperienceInCurrentDomain"] = st.number_input(
        "Experience in Current Field", min_value=0, step=1, value=1
    )
    return data


def load_model():
    return joblib.load(MODEL_PATH)


def predict(data: dict):
    """Transform the data then make predictions"""
    df = pd.DataFrame(data, index=[0])
    df = transform_data(df)
    model = load_model()
    result = model.predict(df)[0]
    return result


def write_predictions(data: dict):
    if st.button("Will this employee leave in 2 years?"):

        prediction = predict(data)
        if prediction == 0:
            st.write("This employee is predicted stay more than two years.")
        else:
            st.write("This employee is predicted to leave in two years.")

def main():
    data = get_inputs()
    write_predictions(data)


if __name__ == "__main__":
    main()
