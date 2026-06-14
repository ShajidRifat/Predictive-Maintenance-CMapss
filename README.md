# Predictive Maintenance of Turbofan Engines using NASA C-MAPSS

A machine learning system for predicting **Remaining Useful Life (RUL)** and classifying **engine health status** using the NASA C-MAPSS benchmark dataset.

This project was developed as an applied Machine Learning and Prognostics & Health Management (PHM) research project focused on aircraft engine degradation analysis and failure prediction.



## Overview

Aircraft engines gradually degrade during operation. Predicting how much useful life remains before failure is critical for reducing maintenance costs, preventing unexpected breakdowns, and improving operational safety.

This project trains machine learning models to:

* Predict Remaining Useful Life (RUL) of turbofan engines
* Classify engine health condition into interpretable categories
* Provide real-time predictions through a Streamlit dashboard
* Support predictive maintenance decision-making



## Features

* Remaining Useful Life (RUL) prediction
* Engine health classification
* Multi-dataset training using FD001–FD004
* Interactive Streamlit dashboard
* Feature scaling and model persistence
* Real-time engine diagnostics
* Automated model training pipeline
* Support for multiple operating conditions and fault modes



## Health Status Classification

Engine health is categorized using the Life Ratio metric.

| Class | Life Ratio  | Status      |
| ----- | ----------- | ----------- |
| 0     | ≤ 0.60      | 🟢 Healthy  |
| 1     | 0.60 – 0.80 | 🟡 Warning  |
| 2     | > 0.80      | 🔴 Critical |



## Dataset

### NASA C-MAPSS Dataset

The project uses the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset, a widely used benchmark for predictive maintenance research.

| Subset | Train Units | Test Units | Operating Conditions | Fault Modes |
| ------ | ----------- | ---------- | -------------------- | ----------- |
| FD001  | 100         | 100        | 1                    | 1           |
| FD002  | 260         | 259        | 6                    | 1           |
| FD003  | 100         | 100        | 1                    | 2           |
| FD004  | 249         | 248        | 6                    | 2           |

The four subsets are merged into a single training dataset to improve model generalization across different operating environments and failure patterns.



## Data Preprocessing

The following preprocessing steps are applied:

* Generate Remaining Useful Life (RUL) labels
* Piecewise linear RUL capping at 125 cycles
* Creation of Life Ratio metric
* Generation of health-status labels
* Feature standardization using StandardScaler
* Dataset merging across FD001–FD004



## Machine Learning Models

### Random Forest Regressor

Used to predict continuous Remaining Useful Life values.

Advantages:

* Handles nonlinear degradation patterns
* Robust against noise
* Performs well on tabular sensor data
* Requires minimal feature engineering

### Random Forest Classifier

Used to classify engine health into:

* Healthy
* Warning
* Critical

### Feature Scaling

A StandardScaler is trained on the combined dataset and saved for inference.



## Project Architecture


NASA C-MAPSS Dataset
          │
          ▼
   Data Preprocessing
          │
          ▼
 Feature Engineering
          │
          ▼
   StandardScaler
          │
          ▼
 ┌─────────────────┬─────────────────┐
 │                 │                 │
 ▼                 ▼
Random Forest   Random Forest
 Regressor       Classifier
 │                 │
 ▼                 ▼
Predicted RUL   Health Status
          │
          ▼
  Streamlit Dashboard


## Project Structure


Predictive-Maintenance-CMapss/
├── CMaps/
│   ├── train_FD001.txt
│   ├── train_FD002.txt
│   ├── train_FD003.txt
│   ├── train_FD004.txt
│   ├── test_FD001.txt
│   ├── test_FD002.txt
│   ├── test_FD003.txt
│   ├── test_FD004.txt
│   └── RUL_FD001.txt ... RUL_FD004.txt
│
├── models/
│   ├── scaler.joblib
│   ├── rf_regressor.joblib     # generated locally — not tracked in git
│   └── rf_classifier.joblib    # generated locally — not tracked in git
│
├── notebooks/
│   ├── classification.ipynb
│   ├── classification_all_data.ipynb
│   ├── reg_class.ipynb
│   ├── reg_class_all.ipynb
│   └── regression_all_data.ipynb
│
├── app.py
├── app_dashboard.py
├── train_and_save_models.py
├── requirements.txt
└── README.md





## Results

| Metric                  | Value       |
| ----------------------- | ----------- |
| RUL Regressor MAE       | YOUR_MAE cycles  |
| RUL Regressor RMSE      | YOUR_RMSE cycles |
| Classification Accuracy | YOUR_ACC%   |
| Classification F1 Score | YOUR_F1%    |

---

## Technologies Used

* Python 3.x
* Scikit-Learn
* Pandas
* NumPy
* Streamlit
* Plotly
* Joblib



## Limitations

* Uses simulated engine data rather than real-world telemetry
* Random Forest does not explicitly model temporal sequences
* Performance may vary on unseen engine types
* Deep learning architectures may achieve better long-term degradation modeling



## Future Improvements

* XGBoost and LightGBM implementation
* LSTM-based RUL prediction
* GRU-based sequence modeling
* SHAP explainability integration
* REST API deployment
* Real-time sensor streaming
* Cloud deployment




