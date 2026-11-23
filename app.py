import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset and model
df = pd.read_csv("campus_train.csv")
df.columns = df.columns.str.strip()

model = joblib.load("model/placement_model.pkl")

# Features
features = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "PlacementTraining",
    "SSC_Marks",
    "HSC_Marks"
]

target = "PlacementStatus"

# Page configuration
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)
# Custom UI Styling

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0e1117, #172338);
}

h1 {
    color: #60a5fa !important;
    font-weight: 800;
}

h2, h3 {
    color: #e2e8f0 !important;
}

div[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    padding: 18px;
    border-radius: 15px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
    border: none;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
}

.stButton > button:hover {
    border: 1px solid #60a5fa;
    color: white;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("🎓 Student Placement Predictor")

st.write(
    "Predict student placement outcomes using Machine Learning."
)

st.divider()

# Input section
st.header("Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1
    )

    internships = st.number_input(
        "Number of Internships",
        min_value=0,
        max_value=10,
        value=1
    )

    projects = st.number_input(
        "Number of Projects",
        min_value=0,
        max_value=20,
        value=2
    )

    workshops = st.number_input(
        "Workshops / Certifications",
        min_value=0,
        max_value=20,
        value=2
    )

    aptitude = st.slider(
        "Aptitude Test Score",
        0, 100, 70
    )

with col2:
    softskills = st.slider(
        "Soft Skills Rating",
        0.0, 5.0, 3.5, step=0.1
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )

    training = st.selectbox(
        "Placement Training",
        ["Yes", "No"]
    )

    ssc = st.slider(
        "SSC Marks (%)",
        0, 100, 75
    )

    hsc = st.slider(
        "HSC Marks (%)",
        0, 100, 75
    )

# Input dataframe
input_data = pd.DataFrame([{
    "CGPA": cgpa,
    "Internships": internships,
    "Projects": projects,
    "Workshops/Certifications": workshops,
    "AptitudeTestScore": aptitude,
    "SoftSkillsRating": softskills,
    "ExtracurricularActivities": extracurricular,
    "PlacementTraining": training,
    "SSC_Marks": ssc,
    "HSC_Marks": hsc
}])

# Prediction
st.divider()

if st.button("Predict Placement", type="primary"):

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    classes = model.classes_

    placed_index = list(classes).index("Placed")

    probability = probabilities[placed_index] * 100

    st.header("Prediction Result")

    if prediction == "Placed":

        st.success("🎉 Predicted: Likely to be Placed")

    else:

        st.error("Predicted: Unlikely to be Placed")

    st.metric(
        "Estimated Placement Probability",
        f"{probability:.2f}%"
    )

    st.progress(int(probability))

    if probability >= 75:
        readiness = "High"

    elif probability >= 50:
        readiness = "Moderate"

    else:
        readiness = "Low"

    st.subheader(f"Placement Readiness: {readiness}")

    # Student Profile Insights

st.subheader("📌 Student Profile Insights")

strengths = []
improvements = []

if cgpa >= 8:
    strengths.append("Strong academic performance (CGPA).")
elif cgpa < 7:
    improvements.append("Try to improve your CGPA.")

if internships >= 2:
    strengths.append("Good internship experience.")
elif internships == 0:
    improvements.append("Gain practical experience through internships.")

if projects >= 3:
    strengths.append("Good project portfolio.")
elif projects < 2:
    improvements.append("Build more practical projects.")

if aptitude >= 75:
    strengths.append("Strong aptitude score.")
elif aptitude < 60:
    improvements.append("Practice aptitude and problem-solving.")

if softskills >= 4:
    strengths.append("Excellent soft skills.")
elif softskills < 3:
    improvements.append("Improve communication and soft skills.")

if workshops >= 3:
    strengths.append("Good workshop/certification participation.")
elif workshops < 2:
    improvements.append("Consider relevant certifications or workshops.")

if training == "Yes":
    strengths.append("Completed placement training.")
else:
    improvements.append("Participate in placement training.")

if extracurricular == "Yes":
    strengths.append("Active in extracurricular activities.")

if ssc >= 80:
    strengths.append("Strong SSC academic record.")

if hsc >= 80:
    strengths.append("Strong HSC academic record.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🟢 Strengths")
    if strengths:
        for item in strengths:
            st.write("✅", item)
    else:
        st.write("Keep developing your profile.")

with col2:
    st.markdown("#### 🟠 Areas to Improve")
    if improvements:
        for item in improvements:
            st.write("📈", item)
    else:
        st.write("Your profile is strong across these selected indicators.")

    st.info(
        "This prediction is generated by a machine learning model "
        "and is intended for educational purposes only."
    )

# Model Performance
st.divider()

st.header("📊 Model Performance")

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="Placed")
recall = recall_score(y_test, y_pred, pos_label="Placed")
f1 = f1_score(y_test, y_pred, pos_label="Placed")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy*100:.2f}%")
c2.metric("Precision", f"{precision*100:.2f}%")
c3.metric("Recall", f"{recall*100:.2f}%")
c4.metric("F1 Score", f"{f1*100:.2f}%")

# Feature importance
st.subheader("Feature Importance")

classifier = model.named_steps["classifier"]

preprocessor = model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

importance = classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature")
)

st.caption(
    "Performance metrics are calculated on the test split. "
    "Predictions are estimates, not guarantees."
)

st.caption("Student Placement Predictor | Machine Learning Project")