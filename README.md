# Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

This project predicts whether a customer is likely to **churn (leave the service)** or **stay** based on customer information.

The project uses Machine Learning techniques to train a classification model and provides a **Streamlit web application** where users can enter customer details and get a churn prediction with probability.

## 🎯 Objective

The main objective of this project is to:

- Analyze customer churn data
- Preprocess categorical and numerical features
- Train a Machine Learning classification model
- Predict whether a customer is likely to churn
- Display the prediction through an interactive web application

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## 🤖 Machine Learning Model

The project uses:

- **Logistic Regression** as the main prediction model
- **Decision Tree** for comparison
- **StandardScaler** for scaling numerical features
- **One-Hot Encoding** for categorical features

### Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains information about customers such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract
- Monthly Charges
- Total Charges

The target variable is:

**Churn**
- `1` → Customer is likely to churn
- `0` → Customer is likely to stay

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── customer_churn.csv
│
├── notebook/
│   └── Customer_Churn_Prediction.ipynb
│
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
├── .gitignore
└── README.md