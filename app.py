
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="ML Model Comparison App", layout="wide")

st.title("Machine Learning Classification Models Comparison")

st.write("Upload a CSV dataset to test trained models.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    X = df.iloc[:, :-1]

    # Convert categorical
    X = pd.get_dummies(X, drop_first=True)

    model_path = "model"

    if os.path.exists(model_path):

        results = {}

        for file in os.listdir(model_path):

            if file.endswith(".pkl"):

                model_name = file.replace(".pkl", "")

                model = joblib.load(os.path.join(model_path, file))

                try:
                    preds = model.predict(X)
                    results[model_name] = preds[:10]
                except:
                    results[model_name] = "Prediction Error"

        st.subheader("Model Predictions (First 10 Rows)")
        st.write(results)

    else:
        st.error("Model folder not found. Upload trained models.")
