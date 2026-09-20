import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict the likelihood of a customer leaving the service.")

st.divider()

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

st.divider()

if st.button("🔍 Predict Churn", type="primary", use_container_width=True):

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "InternetService": [internet_service],
        "Contract": [contract],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )

    for column in feature_columns:
        if column not in input_data.columns:
            input_data[column] = 0

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    numerical_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    input_data[numerical_columns] = scaler.transform(
        input_data[numerical_columns]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if probability >= 0.70:
        risk = "High Risk"
        message = "This customer has a high predicted probability of churn."
    elif probability >= 0.40:
        risk = "Medium Risk"
        message = "This customer has a moderate predicted probability of churn."
    else:
        risk = "Low Risk"
        message = "This customer has a low predicted probability of churn."

    st.divider()

    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        if prediction == 1:
            st.error("⚠️ Likely to CHURN")
        else:
            st.success("✅ Likely to STAY")

    with result_col2:
        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    with result_col3:
        st.metric(
            "Risk Level",
            risk
        )

    st.divider()

    graph_col, summary_col = st.columns(2)

    with graph_col:

        st.subheader("Churn Probability")

        chart_data = pd.DataFrame({
            "Status": ["Stay", "Churn"],
            "Probability": [
                1 - probability,
                probability
            ]
        })

        st.bar_chart(
            chart_data.set_index("Status")
        )

    with summary_col:

        st.subheader("Customer Summary")

        st.write(f"**Gender:** {gender}")
        st.write(f"**Tenure:** {tenure} months")
        st.write(f"**Contract:** {contract}")
        st.write(f"**Internet Service:** {internet_service}")
        st.write(f"**Monthly Charges:** {monthly_charges:.2f}")
        st.write(f"**Total Charges:** {total_charges:.2f}")

    st.divider()

    st.subheader("Risk Assessment")

    if risk == "High Risk":
        st.warning(
            "⚠️ " + message +
            " Consider reviewing this customer's service experience, "
            "contract, and charges."
        )

    elif risk == "Medium Risk":
        st.info(
            "ℹ️ " + message +
            " This customer may benefit from additional engagement."
        )

    else:
        st.success(
            "✅ " + message +
            " The model predicts a relatively low churn probability."
        )

    st.divider()

    st.subheader("Prediction Details")

    detail_data = pd.DataFrame({
        "Prediction": [
            "Churn" if prediction == 1 else "Stay"
        ],
        "Probability": [
            f"{probability:.2%}"
        ],
        "Risk Level": [
            risk
        ]
    })

    st.table(detail_data)