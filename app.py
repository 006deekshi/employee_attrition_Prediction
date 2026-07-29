import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👔", layout="wide")

# ── Train model + SHAP explainer once and cache ─────────────────────────────
@st.cache_resource
def load_model():
    dataset = pd.read_csv("processed table.csv")
    dataset.drop(columns=["Unnamed: 0"], inplace=True)
    y = dataset["Attrition"]
    x = dataset.drop("Attrition", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(C=0.1, random_state=42, solver="liblinear")
    model.fit(x_train, y_train)
    explainer = shap.LinearExplainer(model, x_train, feature_perturbation="interventional")
    return model, x.columns.tolist(), explainer, x_train


model, feature_cols, explainer, x_train = load_model()

# ── Encoding maps (from preprocessing notebook) ─────────────────────────────
BUSINESS_TRAVEL = {"Non-Travel": 2, "Travel_Rarely": 0, "Travel_Frequently": 1}
DEPARTMENT = {"Research & Development": 0, "Sales": 1, "Human Resources": 2}
EDUCATION_FIELD = {
    "Life Sciences": 5, "Other": 1, "Medical": 4,
    "Marketing": 3, "Technical Degree": 0, "Human Resources": 2,
}
GENDER = {"Male": 1, "Female": 0}
JOB_ROLE = {
    "Sales Executive": 8, "Research Scientist": 7, "Laboratory Technician": 6,
    "Manufacturing Director": 5, "Healthcare Representative": 4,
    "Manager": 3, "Sales Representative": 2, "Research Director": 1,
    "Human Resources": 0,
}
MARITAL_STATUS = {"Single": 0, "Married": 2, "Divorced": 1}
OVERTIME = {"Yes": 1, "No": 0}
ATTRITION_LABEL = {0: "No", 1: "Yes"}

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("👔 Employee Attrition Predictor")
st.markdown("Fill in the employee details below to predict the likelihood of attrition.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    age = st.slider("Age", 18, 60, 35)
    gender = st.selectbox("Gender", list(GENDER.keys()))
    marital_status = st.selectbox("Marital Status", list(MARITAL_STATUS.keys()))
    education = st.selectbox("Education Level (1=Below College … 5=Doctor)", [1, 2, 3, 4, 5])
    education_field = st.selectbox("Education Field", list(EDUCATION_FIELD.keys()))
    distance_from_home = st.slider("Distance From Home (km)", 1, 29, 5)

with col2:
    st.subheader("Job Info")
    department = st.selectbox("Department", list(DEPARTMENT.keys()))
    job_role = st.selectbox("Job Role", list(JOB_ROLE.keys()))
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    job_involvement = st.selectbox("Job Involvement (1=Low … 4=Very High)", [1, 2, 3, 4])
    job_satisfaction = st.selectbox("Job Satisfaction (1=Low … 4=Very High)", [1, 2, 3, 4])
    overtime = st.selectbox("Over Time", list(OVERTIME.keys()))
    business_travel = st.selectbox("Business Travel", list(BUSINESS_TRAVEL.keys()))

with col3:
    st.subheader("Compensation & Experience")
    monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
    daily_rate = st.slider("Daily Rate", 100, 1500, 800)
    hourly_rate = st.slider("Hourly Rate", 30, 100, 65)
    monthly_rate = st.slider("Monthly Rate", 2000, 27000, 14000)
    percent_salary_hike = st.slider("Percent Salary Hike", 11, 25, 15)
    stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    performance_rating = st.selectbox("Performance Rating (3=Excellent, 4=Outstanding)", [3, 4])

st.markdown("---")
col4, col5 = st.columns(2)

with col4:
    st.subheader("Tenure & Relationships")
    num_companies_worked = st.slider("Num Companies Worked", 0, 9, 2)
    total_working_years = st.slider("Total Working Years", 0, 40, 10)
    years_at_company = st.slider("Years At Company", 0, 40, 5)
    years_in_current_role = st.slider("Years In Current Role", 0, 18, 4)
    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
    years_with_curr_manager = st.slider("Years With Current Manager", 0, 17, 4)
    employee_number = st.number_input("Employee Number", 1, 2100, 100)

with col5:
    st.subheader("Satisfaction & Training")
    environment_satisfaction = st.selectbox("Environment Satisfaction (1=Low … 4=Very High)", [1, 2, 3, 4])
    relationship_satisfaction = st.selectbox("Relationship Satisfaction (1=Low … 4=Very High)", [1, 2, 3, 4])
    work_life_balance = st.selectbox("Work Life Balance (1=Bad … 4=Best)", [1, 2, 3, 4])
    training_times_last_year = st.slider("Training Times Last Year", 0, 6, 3)

st.markdown("---")

if st.button("🔍 Predict Attrition", use_container_width=True):
    input_data = pd.DataFrame([{
        "Age": age,
        "BusinessTravel": BUSINESS_TRAVEL[business_travel],
        "DailyRate": daily_rate,
        "Department": DEPARTMENT[department],
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": EDUCATION_FIELD[education_field],
        "EmployeeNumber": employee_number,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": GENDER[gender],
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": JOB_ROLE[job_role],
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": MARITAL_STATUS[marital_status],
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": OVERTIME[overtime],
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    }])[feature_cols]

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    if prediction == "Yes":
        st.error(f"⚠️ High Attrition Risk  —  Confidence: {proba[1]*100:.1f}%")
    else:
        st.success(f"✅ Low Attrition Risk  —  Confidence: {proba[0]*100:.1f}%")

    # Attrition probability bar
    st.progress(float(proba[1]), text=f"Attrition probability: {proba[1]*100:.1f}%")

    # ── SHAP Explanation ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔎 Why this prediction?")
    st.caption("Red bars push toward **Attrition (Yes)**, blue bars push toward **No Attrition**.")

    shap_values = explainer.shap_values(input_data)

    # LinearExplainer returns list for multi-class or array for binary — handle both
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    # Build a clean DataFrame for display
    shap_df = pd.DataFrame({
        "Feature": feature_cols,
        "Your Value": input_data.iloc[0].values,
        "SHAP Impact": sv,
    })
    shap_df["abs"] = shap_df["SHAP Impact"].abs()
    shap_df = shap_df.sort_values("abs", ascending=False).drop(columns="abs").head(15)

    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_df["SHAP Impact"]]
    ax.barh(shap_df["Feature"][::-1], shap_df["SHAP Impact"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value  (+  = higher attrition risk,  −  = lower risk)")
    ax.set_title("Top Features Driving This Prediction")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Readable table
    st.dataframe(
        shap_df.rename(columns={"SHAP Impact": "Impact  (+ = more attrition risk)"})
               .reset_index(drop=True),
        use_container_width=True,
    )
