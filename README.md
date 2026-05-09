# Stock Prediction 1H

A machine learning project for predicting Apple stock price movement using 1-hour market data.

The project includes data preprocessing, feature engineering, model training, an API layer, and a dashboard structure for visualization and future integration.

---

## Project Overview

This repository was created as a practical machine learning workflow focused on time-series stock market data.

The project explores the complete pipeline from raw financial data preparation to prediction model integration.

Main components of the project include:

- Data preprocessing and feature engineering
- LSTM model training
- Scaler and model serialization
- Backend API structure using FastAPI
- Dashboard integration structure
- Organized project architecture for data, notebooks, reports, and source code

---

## Tech Stack

- Python
- Pandas
- NumPy
- TensorFlow / Keras
- Scikit-learn
- FastAPI
- Flask
- Jupyter Notebook
- Git / GitHub

---

## Repository Structure

```text
stock-prediction-1h/
│
├── aapl_dashboard/
│   ├── templates/
│   ├── app.py
│   └── requirements.txt
│
├── data/
│   ├── processed/
│   └── raw/
│
├── notebooks/
│   ├── 01_load_aapl_data.ipynb
│   ├── 02_make_target_and_features.ipynb
│   ├── lstm_model.keras
│   ├── scaler_X.pkl
│   └── scaler_y.pkl
│
├── reports/
│
├── src/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Main Features

- Uses 1-hour historical AAPL stock market data
- Performs preprocessing and feature engineering for time-series prediction
- Trains an LSTM-based prediction model
- Stores trained model and scalers for reuse
- Includes a FastAPI backend structure
- Includes a dashboard structure for visualization and integration

---

## Setup

Clone the repository:

```bash
git clone https://github.com/sinex-cloud/stock-prediction-1h.git
cd stock-prediction-1h
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the FastAPI backend:

```bash
uvicorn src.app:app --reload
```

Run the dashboard:

```bash
python aapl_dashboard/app.py
```

---

## Future Improvements

- Improve model evaluation and visualization
- Dockerize the application
- Add deployment support
- Improve dashboard interaction
- Extend API functionality
- Integrate cloud deployment workflows

---

## Disclaimer

This project was developed for educational purposes only.

It should not be considered financial advice.
