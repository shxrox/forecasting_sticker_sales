# 📈 Forecasting Sticker Sales Using Deep Learning

A Computational Intelligence project that predicts retail sticker sales using a Deep Learning Multi-Layer Perceptron (MLP). The model was developed for the Kaggle **Forecasting Sticker Sales** competition and deployed as an interactive web application using Streamlit.

**Live Demo:** https://forecasting-sticker-sales.streamlit.app/

---

## Project Overview

Retail sales forecasting is a challenging time-series problem influenced by seasonality, product popularity, store characteristics, and regional demand. This project applies a Deep Learning approach to learn these relationships from historical data and generate sales predictions for future dates.

The project demonstrates the complete machine learning lifecycle, including:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Neural network development
- Model evaluation
- Web application deployment

---

## Technologies Used

- **Python 3.11**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Streamlit**

---

## Machine Learning Pipeline

The project follows the standard machine learning workflow:

### 1. Data Preparation
- Load the Kaggle training dataset.
- Remove records containing missing target values.
- Explore trends, seasonality, and product distributions.

### 2. Feature Engineering
The following features were created:

- Year
- Month
- Day
- Day of Week
- Weekend indicator

Calendar variables were transformed using sine and cosine encoding to preserve their cyclical nature.

Categorical variables (Country, Store, and Product) were converted using One-Hot Encoding.

Finally, numerical features were standardised using `StandardScaler`.

### 3. Model Development

A Multi-Layer Perceptron (MLP) was implemented using TensorFlow/Keras.

Architecture:

- Input Layer
- Dense Layer (128 neurons, ReLU)
- Dense Layer (64 neurons, ReLU)
- Dense Layer (32 neurons, ReLU)
- Output Layer (1 neuron)

The model was trained using:

- Adam Optimizer
- Mean Squared Error (MSE) Loss
- Batch Size: 256
- Epochs: 30
- Dropout Rate: 20%

---

## Model Evaluation

The trained model was evaluated using both local validation and the official Kaggle leaderboard.

**Kaggle Results**

- Public Leaderboard Score: **1.30995**
- Private Leaderboard Score: **1.81006**

These results demonstrate that the model generalises well to unseen data and can successfully capture seasonal sales patterns.

---

## Web Application

The trained model was deployed using Streamlit, allowing users to generate predictions without retraining the neural network.

Users can select:

- Forecast Date
- Country
- Store
- Product

The application automatically applies the same preprocessing pipeline used during training before generating a prediction.

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/shxrox/forecasting_sticker_sales.git
cd forecasting_sticker_sales
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Author

**Sharon Devasudan**

---

This project was developed as part of the **CIS6005 – Computational Intelligence** module and demonstrates the application of Deep Learning techniques to a real-world retail sales forecasting problem.
