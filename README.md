🎓 Student Placement Predictor using Machine Learning

📌 Project Overview

Student Placement Predictor is a machine learning-based web application that predicts the placement probability of students based on their academic performance, technical skills, internships, projects, aptitude scores, and other relevant factors.

The project aims to help students understand their placement prospects, identify areas for improvement, and prepare strategically for campus recruitment.

🌐 Live Demo

https://student-placement-predictor-bzqzbnbrmk7kwsmvaqcnpn.streamlit.app/

✨ Key Features

- Predicts whether a student is likely to be Placed or Not Placed.
- Displays placement probability.
- Provides student readiness assessment.
- Identifies strengths and areas requiring improvement.
- Displays model evaluation metrics.
- Visualizes important features influencing predictions.

🛠️ Tech Stack

- Programming Language: Python
- Frontend: Streamlit
- Machine Learning: Scikit-learn
- Algorithm: Random Forest Classifier
- Data Processing: Pandas, NumPy
- Data Visualization: Plotly
- Model Serialization: Joblib
- Development Environment: VS Code
- Version Control: Git and GitHub
- Deployment: Streamlit Cloud

📊 Dataset and Input Features

The model uses the following student-related features:

1. CGPA
2. Number of Internships
3. Number of Projects
4. Workshops/Certifications
5. Aptitude Test Score
6. Soft Skills Rating
7. Extracurricular Activities
8. Placement Training
9. SSC Marks
10. HSC Marks

Target Variable: Placement Status (Placed / NotPlaced)

⚙️ How It Works

1. Student enters their academic and skill-related information.
2. Input data is processed by the trained machine learning pipeline.
3. Random Forest Classifier analyzes the input features.
4. The application predicts placement status and placement probability.
5. The system displays readiness insights, strengths, and improvement areas.

🚀 Run the Project Locally

Step 1: Clone the Repository

git clone https://github.com/Rahul-Coder-For-Fun/student-placement-predictor.git

Step 2: Open the Project Folder

cd student-placement-predictor

Step 3: Install Dependencies

pip install -r requirements.txt

Step 4: Train the Model

python train_model.py

Step 5: Run the Application

streamlit run app.py

📁 Project Structure

student-placement-predictor/
│
├── app.py
├── train_model.py
├── campus_train.csv
├── requirements.txt
├── README.md
│
└── model/
    └── placement_model.pkl

📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

🎯 Future Enhancements

- Use larger and more diverse datasets.
- Improve prediction accuracy and model generalization.
- Add personalized learning recommendations.
- Introduce student performance tracking.
- Enhance the user interface and analytics dashboard.

👨‍💻 Developer

GitHub: https://github.com/Rahul-Coder-For-Fun

⚠️ Disclaimer

This application provides predictions based on available dataset patterns. It is intended for educational and guidance purposes and does not guarantee actual placement outcomes.
