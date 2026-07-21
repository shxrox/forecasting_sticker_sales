import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Sticker Sales Forecasting System",
    layout="centered"
)


def engineer_features(df):
    df = df.copy()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)

    df = pd.get_dummies(
        df,
        columns=['country', 'store', 'product'],
        drop_first=False
    )

    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    return df


@st.cache_resource
def load_system_artifacts():

    model = tf.keras.models.load_model(
        'sticker_sales_model.h5',
        compile=False
    )

    scaler = joblib.load(
        'feature_scaler.pkl'
    )

    train_df = pd.read_csv(
        'train.csv',
        nrows=100
    )

    train_df['date'] = pd.to_datetime(
        train_df['date']
    )

    return model, scaler, train_df


try:
    model, scaler, train_df = load_system_artifacts()

except Exception as e:
    st.error(
        f"Error loading system files: {e}. "
        "Ensure model, scaler, and train.csv are in the correct directory."
    )
    st.stop()


st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 35px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">Sticker Sales Forecasting System</div>',
    unsafe_allow_html=True
)


st.subheader("Forecast Settings")


selected_date = st.date_input(
    "Forecast Date",
    datetime(2026, 1, 1)
)


selected_country = st.selectbox(
    "Country",
    sorted(train_df['country'].unique())
)


selected_store = st.selectbox(
    "Store",
    sorted(train_df['store'].unique())
)


selected_product = st.selectbox(
    "Product",
    sorted(train_df['product'].unique())
)


st.write("")


if st.button(
    "Generate Forecast",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame(
        {
            'date': [pd.to_datetime(selected_date)],
            'country': [selected_country],
            'store': [selected_store],
            'product': [selected_product]
        }
    )


    with st.spinner(
        "Processing prediction..."
    ):

        train_processed = engineer_features(
            train_df
        )

        input_processed = engineer_features(
            input_data
        )


        _, input_aligned = train_processed.align(
            input_processed,
            join='left',
            axis=1,
            fill_value=0
        )


        for col in [
            'id',
            'date',
            'num_sold'
        ]:
            if col in input_aligned.columns:
                input_aligned = input_aligned.drop(
                    col,
                    axis=1
                )


        input_aligned = input_aligned[
            scaler.feature_names_in_
        ]


        features_scaled = scaler.transform(
            input_aligned
        )


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


    st.success(
        "Forecast generated successfully"
    )


    st.subheader("Forecast Result")


    col1, col2, col3 = st.columns(3)


    with col1:
        st.metric(
            "Predicted Units Sold",
            f"{final_prediction:,}"
        )


    with col2:
        st.metric(
            "Country",
            selected_country
        )


    with col3:
        st.metric(
            "Product",
            selected_product
        )