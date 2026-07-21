import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Sticker Sales Forecaster",
    layout="centered",
    page_icon="📊"
)


def engineer_features_inference(input_df, train_df):
    df = input_df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    combined = pd.concat(
        [
            train_df[["country", "store", "product"]],
            df[["country", "store", "product"]]
        ],
        ignore_index=True
    )

    combined_encoded = pd.get_dummies(
        combined,
        columns=["country", "store", "product"]
    )

    df_encoded = combined_encoded.iloc[-1:].copy()

    numerical_features = [
        "year",
        "month",
        "day",
        "dayofweek",
        "is_weekend",
        "month_sin",
        "month_cos",
        "dayofweek_sin",
        "dayofweek_cos"
    ]

    for col in numerical_features:
        df_encoded[col] = df[col].values[0]

    scaler_features = scaler.feature_names_in_

    for col in scaler_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[scaler_features]

    return df_encoded


@st.cache_resource
def load_system_artifacts():
    model = tf.keras.models.load_model(
        "sticker_sales_model.h5",
        compile=False
    )

    scaler = joblib.load("feature_scaler.pkl")

    train_df = pd.read_csv("train.csv")

    train_df["date"] = pd.to_datetime(train_df["date"])

    return model, scaler, train_df


try:
    model, scaler, train_df = load_system_artifacts()

except Exception as e:
    st.error(
        f"Error loading system files: {e}. "
        "Ensure model, scaler, and train.csv are in the correct directory."
    )
    st.stop()


st.title("📊 Sticker Sales Forecasting App")

st.markdown(
    "This application predicts future sticker sales using a trained Deep Learning model."
)


col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Select Date",
        datetime(2026, 1, 1)
    )

    selected_country = st.selectbox(
        "Select Country",
        sorted(train_df["country"].unique())
    )


with col2:
    selected_store = st.selectbox(
        "Select Store",
        sorted(train_df["store"].unique())
    )

    selected_product = st.selectbox(
        "Select Product",
        sorted(train_df["product"].unique())
    )


if st.button("Predict Sales", type="primary"):

    input_data = pd.DataFrame(
        {
            "date": [pd.to_datetime(selected_date)],
            "country": [selected_country],
            "store": [selected_store],
            "product": [selected_product]
        }
    )

    with st.spinner("Calculating forecast using deep learning..."):

        features = engineer_features_inference(
            input_data,
            train_df
        )

        features_scaled = scaler.transform(features)

        prediction = model.predict(
            features_scaled,
            verbose=0
        )

        final_prediction = int(
            max(
                0,
                np.round(prediction[0][0])
            )
        )

        st.success("Prediction Complete!")

        st.metric(
            label="Predicted Units Sold",
            value=final_prediction
        )