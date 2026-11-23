import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("campus_train.csv")

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

# Input features
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

# Target variable
target = "PlacementStatus"

X = df[features]
y = df[target]

# Categorical and numerical columns
categorical_features = [
    "ExtracurricularActivities",
    "PlacementTraining"
]

numerical_features = [
    col for col in features
    if col not in categorical_features
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# Machine Learning model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed!")
print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)

joblib.dump(pipeline, "model/placement_model.pkl")

print("\nModel saved successfully!")