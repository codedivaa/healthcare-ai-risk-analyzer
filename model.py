from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import numpy as np


def train_model(df):

    cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for col in cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        )
    }

    best_model = None
    best_accuracy = 0
    best_cm = None
    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )
        results[name] = accuracy

        print(f"{name}: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_cm = confusion_matrix(
                y_test,
                predictions
            )

    return (
        best_model,
        best_accuracy,
        feature_names,
        best_cm,
        results
    )