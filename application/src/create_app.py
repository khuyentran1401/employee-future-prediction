import json

import requests
import streamlit as st


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


def write_predictions(data: dict):
    if st.button("Will this employee leave in 2 years?"):
        data_json = json.dumps(data)

        prediction = requests.post(
            "https://employee-predict-1.herokuapp.com/predict",
            headers={"content-type": "application/json"},
            data=data_json,
        ).text[0]

        if prediction == "0":
            st.write("This employee is predicted stay more than two years.")
        else:
            st.write("This employee is predicted to leave in two years.")


def main():
    data = get_inputs()
    write_predictions(data)


if __name__ == "__main__":
    main()
