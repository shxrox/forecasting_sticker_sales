import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime

# Configure the Streamlit page layout
st.set_page_config(page_title="Sticker Sales Forecaster", layout="centered", page_icon="📊")

# Replicate the feature engineering logic used during model training
def engineer_features_inference(input_df, train_df_for_alignment):
    df = input_df.copy()
    
    # Generate time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    
    # Combine with a snippet of train_df to generate the exact same one-hot encoded structure
    combined = pd.concat([train_df_for_alignment[['country', 'store', 'product']], df[['country', 'store', 'product']]])
    combined_dummies = pd.get_dummies(combined, columns=['country', 'store', 'product'])
    
    # Extract just the newly encoded inference row
    df_encoded = combined_dummies.iloc[-1:].copy()
    
    # Insert the numerical features back into the encoded dataframe
    numerical_cols = ['year', 'month', 'day', 'dayofweek', 'is_weekend', 
                      'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos']
    
    for col in numerical_cols:
        df_encoded[col] = df[col].values
        
    # Reorder columns to strictly match the training data scaler expectations
    train_engineered = pd.get_dummies(train_df_for_alignment, columns=['country', 'store', 'product'])
    expected_cols = [c for c in train_engineered.columns if c not in ['id', 'date', 'num_sold']]
    
    # Ensure any boolean columns are converted to integers
    df_encoded = df_encoded[expected_cols]
    for col in df_encoded.select_dtypes(include=['bool']).columns:
        df_encoded[col] = df_encoded[col].astype(int)
        
    return df_encoded

# Cache the heavy system resources so the app doesn't reload them on every button click
@st.cache_resource
def load_system_artifacts():
    model = tf.keras.models.load_model('sticker_sales_model.h5')
    scaler = joblib.load('feature_scaler.pkl')
    # Load a small subset of training data to extract structural categories
    train_df = pd.read_csv('train.csv', nrows=100) 
    return model, scaler, train_df

try:
    model, scaler, train_df = load_system_artifacts()
except Exception as e:
    st.error(f"Error loading system files: {e}. Ensure model, scaler, and train.csv are in the correct directory.")
    st.stop()

# --- Application UI ---
st.title("📊 Sticker Sales Forecasting App")
st.markdown("This application predicts future sticker sales based on a trained Deep Learning model.")

# Create a clean 2-column layout for inputs
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("Select Date", datetime(2026, 1, 1))
    selected_country = st.selectbox("Select Country", train_df['country'].unique())
with col2:
    selected_store = st.selectbox("Select Store", train_df['store'].unique())
    selected_product = st.selectbox("Select Product", train_df['product'].unique())

# Prediction Trigger
if st.button("Predict Sales", type="primary"):
    input_data = pd.DataFrame({
        'date': [pd.to_datetime(selected_date)],
        'country': [selected_country],
        'store': [selected_store],
        'product': [selected_product]
    })
    
    with st.spinner('Calculating forecast using deep learning...'):
        # Pass through the exact pipeline used in training
        features = engineer_features_inference(input_data, train_df)
        features_scaled = scaler.transform(features)
        
        # Generate prediction
        prediction = model.predict(features_scaled, verbose=0)
        
        # Round the output and ensure we don't predict negative sales
        final_prediction = int(max(0, np.round(prediction[0][0])))
        
        st.success("Prediction Complete!")
        st.metric(label="Predicted Units Sold", value=final_prediction)