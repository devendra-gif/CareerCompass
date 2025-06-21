import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and label encoders
model = pickle.load(open("career_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# List of features expected by the model
feature_columns = [
    'Age', 'Gender', 'High_School_GPA', 'SAT_Score', 'University_Ranking',
    'University_GPA', 'Internships_Completed', 'Projects_Completed',
    'Certifications', 'Soft_Skills_Score', 'Networking_Score', 'Job_Offers',
    'Starting_Salary', 'Career_Satisfaction', 'Years_to_Promotion',
    'Current_Job_Level', 'Work_Life_Balance', 'Entrepreneurship'
]

st.title("CareerCompass: Tech vs Non-Tech Career Predictor")
st.markdown("### Enter your information below:")

# Collect inputs from user
age = st.slider("Age", 16, 60, 22)
gender = st.selectbox("Gender", label_encoders["Gender"].classes_)
gpa = st.slider("High School GPA", 0.0, 4.0, 3.0)
sat = st.slider("SAT Score", 400, 1600, 1000)
ranking = st.slider("University Ranking (1 = Top)", 1, 500, 100)
uni_gpa = st.slider("University GPA", 0.0, 4.0, 3.0)
interns = st.slider("Internships Completed", 0, 10, 1)
projects = st.slider("Projects Completed", 0, 20, 3)
certs = st.slider("Certifications", 0, 10, 1)
soft_skills = st.slider("Soft Skills Score (1-5)", 1, 5, 3)
networking = st.slider("Networking Score (1-5)", 1, 5, 3)
offers = st.slider("Job Offers Received", 0, 5, 1)
salary = st.slider("Starting Salary (in thousands)", 0, 200, 50)
satisfaction = st.slider("Career Satisfaction (1-5)", 1, 5, 3)
promotion_years = st.slider("Years to Promotion", 0, 10, 2)
job_level = st.selectbox("Current Job Level", label_encoders["Current_Job_Level"].classes_)
balance = st.slider("Work-Life Balance (1-5)", 1, 5, 3)
entrepreneurship = st.selectbox("Entrepreneurship Interest", label_encoders["Entrepreneurship"].classes_)

# Prepare input dictionary
input_dict = {
    'Age': age,
    'Gender': gender,
    'High_School_GPA': gpa,
    'SAT_Score': sat,
    'University_Ranking': ranking,
    'University_GPA': uni_gpa,
    'Internships_Completed': interns,
    'Projects_Completed': projects,
    'Certifications': certs,
    'Soft_Skills_Score': soft_skills,
    'Networking_Score': networking,
    'Job_Offers': offers,
    'Starting_Salary': salary,
    'Career_Satisfaction': satisfaction,
    'Years_to_Promotion': promotion_years,
    'Current_Job_Level': job_level,
    'Work_Life_Balance': balance,
    'Entrepreneurship': entrepreneurship
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Apply label encoding to categorical columns
for col in input_df.columns:
    if col in label_encoders and input_df[col].dtype == object:
        le = label_encoders[col]
        value = input_df[col][0]
        if value not in le.classes_:
            st.error(f"'{value}' is not a recognized value for '{col}'")
            st.stop()
        input_df[col] = le.transform([value])

# Predict
if st.button("Predict Career Path"):
    prediction = model.predict(input_df)[0]
    result = "💻 Tech" if prediction == 1 else "📘 Non-Tech"
    st.success(f"Recommended Career Path: **{result}**")