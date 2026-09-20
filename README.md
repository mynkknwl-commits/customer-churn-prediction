# Customer Churn Prediction Using Machine Learning

## Project Overview

This project predicts whether a customer is likely to churn or stay using Machine Learning.

The project uses the Telco Customer Churn dataset and a Logistic Regression classification model. A Streamlit web application allows users to enter customer details and receive a churn prediction.

## Objective

- Analyze customer churn data
- Preprocess the data
- Train a Machine Learning classification model
- Predict customer churn
- Display predictions through a Streamlit web application

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Git & GitHub

## Machine Learning

The main model used is Logistic Regression.

A Decision Tree model was also used for comparison.

The project uses One-Hot Encoding for categorical features and StandardScaler for numerical features.

## Dataset

The project uses the Telco Customer Churn dataset.

The target variable is:

- 1 ? Customer is likely to churn
- 0 ? Customer is likely to stay

## Project Structure

- data/
- notebook/
- app.py
- churn_model.pkl
- scaler.pkl
- feature_columns.pkl
- requirements.txt
- .gitignore
- README.md

## How to Run

Clone the repository:

    git clone https://github.com/mynkknwl-commits/customer-churn-prediction.git

Open the project:

    cd customer-churn-prediction

Install dependencies:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

## Application

The Streamlit application provides:

- Customer churn prediction
- Churn probability

## Learning Outcomes

- Data preprocessing
- Machine Learning classification
- Feature encoding
- Feature scaling
- Model training
- Model saving and loading
- Streamlit
- Git and GitHub
- Deployment

## Author

Mayank Singh Kanwal

B.Sc. Information Technology Student
