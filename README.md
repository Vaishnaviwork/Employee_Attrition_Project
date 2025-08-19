🚀 Employee Attrition Analysis and Prediction  

📌 Project Overview  
Employee turnover poses a significant challenge for organizations, leading to increased costs, reduced productivity, and disruption in teams.  
This project aims to analyze employee data, identify key drivers of attrition, and predict employees at risk using machine learning.  

The project includes:  
✔️ Data Preprocessing & Cleaning  
✔️ Exploratory Data Analysis (EDA)  
✔️ Feature Engineering  
✔️ Machine Learning Model Development & Evaluation  
✔️ Interactive Streamlit Dashboard for prediction and insights  


🏢 Domain – HR Analytics  
Goal: Predict and prevent employee attrition by providing actionable insights to HR teams.  

🔑 Business Use Cases  
- Employee Retention: Identify at-risk employees and take preventive actions.  
- Cost Optimization: Reduce hiring, training, and onboarding costs.  
- Workforce Planning: Align retention strategies with organizational goals.  

---

⚙️ Approach  

1. Data Collection & Preprocessing  
   - Handle missing values, outliers, and categorical encoding.  
2. Exploratory Data Analysis (EDA)  
   - Identify attrition trends, correlations, and patterns.  
3. Feature Engineering  
   - Derived features like tenure categories, engagement scores, etc.  
4. Model Building  
   - Logistic Regression, Random Forest, and other ML algorithms.  
5. Model Evaluation  
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.  
6. Deployment  
   - Interactive Streamlit App with dashboards and prediction form.  

---

📊 Results  

- Predictive Model Accuracy: Achieved >85% on test data.  
- Key Drivers Identified: Job Satisfaction, Overtime, Monthly Income, Years at Company, Work-Life Balance.  
- At-Risk Employees: Ranked list of employees with highest probability of leaving.  
- Actionable Insights: Suggested retention strategies (bonuses, promotions, training, etc.).  
- Streamlit Dashboard: Interactive visualization of attrition patterns and real-time prediction tool.  

---

📈 Evaluation Metrics  

- Accuracy – Overall correctness of predictions.  
- Precision & Recall – Balance false positives & false negatives.  
- F1-Score – Harmonic mean of Precision & Recall.  
- AUC-ROC – Model’s ability to separate attrition vs non-attrition.  
- Confusion Matrix – Performance visualization.  

---
 🛠️ Tech Stack  

- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)  
- EDA & Visualization: Matplotlib, Seaborn, Plotly  
- ML Algorithms: Logistic Regression, Decision Trees, Random Forests  
- App Deployment: Streamlit  
- Model Saving: Pickle / Joblib  

---
Project file structure

├── Employee_dataclean.ipynb # Data Cleaning & Preprocessing
├── Employeeapp.py # Streamlit Dashboard
├── employee_cleaned.csv # Cleaned dataset
├── employee_features.csv # Engineered features dataset
├── README.md # Project Documentation


---

## 📑 Dataset  

The dataset contains 35 features such as demographics, salary, work experience, and satisfaction metrics.  

Example features:  
- Age, Department, JobRole, MonthlyIncome, JobSatisfaction, OverTime, YearsAtCompany  
- Target Variable: Attrition (0 = Stayed, 1 = Left)  

📂 Dataset Source: (https://www.kaggle.com/code/varunsaikanuri/employee-attrition-analysis) 

---

## 🚀 Running the Project  

1. Clone the repo:  
   ```bash
   git clone https://github.com/yourusername/Employee-Attrition-Prediction.git
   cd Employee-Attrition-Prediction

Run the Streamlit app:

streamlit run Employeeapp.py

📸 Streamlit Dashboard Preview

👉 Visual Insights: Attrition heatmap, job role analysis, income distribution, satisfaction trends.
👉 Prediction Tool: Enter employee details and get probability of attrition (%).

📌 Future Enhancements

Add SHAP explainability for feature importance at individual predictions.

Deploy app on Streamlit Cloud / Heroku / AWS.

Enhance with deep learning models for improved accuracy.

Integrate with real-time HR systems.

## 📂 Project Structure  

