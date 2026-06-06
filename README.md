# 🏥 AI Health Risk Analyzer

An interactive Machine Learning application that predicts diabetes risk using patient health metrics and provides explainable health insights through an intuitive Streamlit dashboard.

---

## 📌 Overview

Healthcare professionals and patients often struggle to quickly interpret health indicators and understand potential diabetes risk.

This project uses Machine Learning to analyze patient health data and estimate diabetes risk based on clinical indicators such as:

* Glucose Level
* BMI
* Blood Pressure
* Age
* Insulin
* Skin Thickness
* Diabetes Pedigree Function
* Pregnancies

The system compares multiple ML algorithms, identifies the best-performing model, and provides risk predictions with visual explanations.

---

## 🚀 Features

### Data Upload

Upload patient datasets in CSV format.

### Data Exploration

* Dataset preview
* Statistical summaries
* Health analytics dashboard
* Interactive visualizations

### Machine Learning

Model comparison using:

* Random Forest Classifier
* Logistic Regression
* Decision Tree Classifier

### Model Evaluation

* Accuracy Score
* Confusion Matrix
* Best Model Selection

### Risk Prediction

Predict diabetes risk for individual patients using:

* Age
* BMI
* Blood Pressure
* Glucose
* Insulin
* Skin Thickness
* Diabetes Pedigree Function
* Pregnancies

### Explainable AI

Provides explanations for predictions based on:

* Elevated Glucose
* High BMI
* Blood Pressure
* Age-related risk factors

---

## 🧠 Machine Learning Pipeline

### Data Cleaning

Missing or invalid values are handled by:

* Replacing zero values in medical fields
* Median value imputation

### Data Splitting

Training and testing datasets are created using:

* Train/Test Split
* Stratified Sampling

### Models Evaluated

1. Random Forest
2. Logistic Regression
3. Decision Tree

### Evaluation Metrics

* Accuracy Score
* Confusion Matrix

The highest-performing model is automatically selected for predictions.

---

## 🏗 System Architecture

Patient Dataset (CSV)
↓
Pandas Data Processing
↓
Data Cleaning & Imputation
↓
Train/Test Split
↓
Model Training
├── Random Forest
├── Logistic Regression
└── Decision Tree
↓
Model Evaluation
↓
Best Model Selection
↓
Risk Prediction
↓
Visualization Dashboard
↓
Health Recommendations

---

## 🛠 Tech Stack

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn

### Visualization

* Matplotlib
* Plotly

### Web Application

* Streamlit

---

## 📊 Dataset

The project is designed for the Pima Indians Diabetes Dataset.

Features include:

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

Target Variable:

* Outcome

---

## 📈 Current Performance

Model accuracy may vary depending on dataset split.

Typical results:

| Model               | Accuracy |
| ------------------- | -------- |
| Random Forest       | 75–82%   |
| Logistic Regression | 74–81%   |
| Decision Tree       | 68–75%   |

---

## ▶ Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start application:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8503
```

---

## 🎯 Future Improvements

* PDF Health Reports
* AI Health Assistant
* Historical Patient Tracking
* Risk Trend Analysis
* Advanced Ensemble Models
* XGBoost Integration
* Cloud Deployment

---

## 👩‍💻 Author

Built using Python, Machine Learning, Pandas, NumPy, Scikit-Learn, Streamlit, Matplotlib and Plotly.

This project was developed as a practical healthcare analytics and risk prediction platform.
