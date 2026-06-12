# 📈 Tesla Stock Price Prediction Using Deep Learning

## Overview

This project predicts Tesla (TSLA) stock closing prices using three Deep Learning architectures:

* SimpleRNN (Simple Recurrent Neural Network)
* GRU (Gated Recurrent Unit)
* LSTM (Long Short-Term Memory)

The models are trained on historical Tesla stock data and evaluated across multiple forecasting horizons (1-day, 5-day, and 10-day predictions). A Streamlit web application is developed for interactive forecasting and model comparison.

---

## Objectives

* Analyze Tesla stock price trends.
* Build and compare multiple recurrent neural network architectures.
* Evaluate forecasting performance using RMSE.
* Deploy the trained models using Streamlit.
* Identify the best-performing architecture for stock price prediction.

---

## Technologies Used

### Programming Language

* Python

### Libraries

* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Streamlit
* Pickle

---

## Project Workflow

### 1. Data Collection

Historical Tesla stock data (TSLA.csv) containing:

* Date
* Open
* High
* Low
* Close
* Volume

### 2. Data Preprocessing

* Missing value handling
* Date formatting
* Feature scaling using MinMaxScaler
* Creation of 60-day sliding windows

### 3. Model Development

Three Deep Learning models were implemented:

#### SimpleRNN

Basic recurrent neural network architecture used as a baseline model.

#### GRU

Gated Recurrent Unit capable of capturing temporal dependencies with fewer parameters than LSTM.

#### LSTM

Long Short-Term Memory network designed to learn long-range dependencies in sequential data.

### 4. Hyperparameter Tuning

GridSearchCV was applied to optimize LSTM hyperparameters:

* Number of Units
* Dropout Rate
* Learning Rate

Best Parameters:

* Units: 64
* Dropout Rate: 0.2
* Learning Rate: 0.001

---

## Model Performance

### RMSE Comparison

| Forecast Horizon | SimpleRNN | GRU   | LSTM      |
| ---------------- | --------- | ----- | --------- |
| 1 Day            | 23.82     | 20.73 | **16.02** |
| 5 Day            | 47.25     | 33.09 | **28.57** |
| 10 Day           | 70.11     | 43.54 | **40.15** |

### Best Model

🏆 **LSTM**

LSTM achieved the lowest RMSE across all forecasting horizons and was selected as the final deployed forecasting model.

---

## Streamlit Dashboard Features

* Historical Tesla stock visualization
* 30-Day Moving Average
* Multi-horizon forecasting (1, 5, 10 days)
* SimpleRNN predictions
* GRU predictions
* LSTM predictions
* Forecast comparison charts
* RMSE comparison table
* Model performance summary
* Raw dataset viewer

---

## Project Structure

```text
Tesla-Stock-Price-Predictor/
│
├── app.py
├── Tesla_Stock_Price_Prediction.ipynb
├── tesla_simplernn_model.keras
├── tesla_gru_model.keras
├── tesla_lstm_model.keras
├── scaler.pkl
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/5rikanth/Tesla-Stock-Price-Predictor.git
cd Tesla-Stock-Price-Predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Results

The comparative analysis demonstrated that:

* SimpleRNN provides baseline forecasting performance.
* GRU significantly improves prediction accuracy.
* LSTM consistently achieves the lowest RMSE across all forecasting horizons.

Therefore, LSTM was selected as the most suitable architecture for Tesla stock price forecasting.

---

## Future Enhancements

* Integration of real-time stock market data
* News sentiment analysis
* Transformer-based forecasting models
* Ensemble learning approaches
* Multi-stock prediction system

---

