import streamlit as st
import joblib
import numpy as np
model = joblib.load(
    "fraud_model.pkl"
)

st.title(
    "Credit Card Fraud Detection"
)

time = st.number_input(
    "Transaction Time"
)

amount = st.number_input(
    "Transaction Amount"
)

if st.button("Predict"):

    sample = np.zeros(30)

    sample[0] = time
    sample[-1] = amount

    prediction = model.predict(
        [sample]
    )

    if prediction[0] == 1:
        st.error(
            "Fraud Transaction"
        )
    else:
        st.success(
            "Genuine Transaction"
        )