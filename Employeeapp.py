import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Load Model and Data
# -------------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

df = pd.read_csv("employee.csv")

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="🤖 Employee Attrition Prediction", layout="wide")

st.title("🤖 Employee Attrition & Insights Dashboard")

menu = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", "📊 Exploratory Data Analysis (EDA)", "🔮 Attrition Prediction", "🚀 Promotion Likelihood"]
)

# -------------------------------
# HOME PAGE
# -------------------------------
if menu == "🏠 Home":
    st.markdown("### Welcome to the **Employee Attrition & Insights App** 🚀")
    st.write(
        """
        This app helps HR teams and managers to:  
        - 📊 Explore workforce insights with interactive EDA  
        - 🔮 Predict employee attrition with probabilities  
        - 🚀 Estimate promotion likelihood based on performance factors  
        """
    )

# -------------------------------
# EDA PAGE
# -------------------------------
elif menu == "📊 Exploratory Data Analysis (EDA)":
    st.subheader("📊 Employee Data Overview")

    # Show dataset preview
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        # Attrition distribution
        fig1 = px.pie(df, names="Attrition", title="Attrition Distribution", hole=0.4,
                      color="Attrition", color_discrete_map={"Yes": "red", "No": "green"})
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Gender split
        fig2 = px.pie(df, names="Gender", title="Gender Split", hole=0.4,
                      color="Gender")
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap
    st.subheader("🔗 Correlation Heatmap")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm", cbar=True)
    st.pyplot(plt)

    # Attrition by JobRole
    st.subheader("👔 Attrition by Job Role")
    fig3 = px.bar(df, x="JobRole", color="Attrition", barmode="group",
                  title="Attrition across Job Roles")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# ATTRITION PREDICTION
# -------------------------------
elif menu == "🔮 Attrition Prediction":
    st.subheader("🔮 Predict Employee Attrition")

    # Select a few important features (instead of all)
    important_features = [
        "Age", "BusinessTravel", "Department", "DistanceFromHome", "Education",
        "EnvironmentSatisfaction", "Gender", "JobRole", "JobSatisfaction",
        "MaritalStatus", "MonthlyIncome", "OverTime", "PercentSalaryHike",
        "TotalWorkingYears", "YearsAtCompany"
    ]

    user_input = {}
    for col in important_features:
        if df[col].dtype == "object":
            user_input[col] = st.selectbox(f"{col}", df[col].unique())
        else:
            user_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    input_df = pd.DataFrame([user_input])

    # Align with model columns
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    if st.button("🔮 Predict Attrition"):
        pred_prob = model.predict_proba(input_df)[0][1] * 100  # Probability of attrition
        pred_class = "Leave" if pred_prob > 50 else "Stay"

        if pred_class == "Leave":
            st.error(f"⚠️ This employee is likely to **Leave**. Probability: {pred_prob:.2f}%")
        else:
            st.success(f"✅ This employee is likely to **Stay**. Probability: {pred_prob:.2f}%")

# -------------------------------
# PROMOTION LIKELIHOOD
# -------------------------------
elif menu == "🚀 Promotion Likelihood":
    st.subheader("🚀 Promotion Likelihood Estimator")

    perf_rating = st.slider("Performance Rating", 1, 4, 3)
    years = st.slider("Years at Company", 0, 40, 5)
    training = st.slider("Trainings Attended Last Year", 0, 10, 2)
    joblevel = st.slider("Job Level", 1, 5, 2)

    # Simple logic for promotion chance
    score = (perf_rating * 0.4) + (training * 0.2) + (years / 40 * 0.2) + (joblevel * 0.2)
    promotion_prob = min(100, score * 25)

    if st.button("🚀 Check Promotion Likelihood"):
        st.info(f"📈 Estimated Promotion Likelihood: **{promotion_prob:.2f}%**")
        if promotion_prob > 75:
            st.success("🎉 High chance of promotion!")
        elif promotion_prob > 50:
            st.warning("⚠️ Moderate chance of promotion.")
        else:
            st.error("❗ Low chance of promotion. Consider improving performance or skills.")