import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import train_model

st.set_page_config(page_title="AI Health Risk Analyzer")

st.title("🏥 AI Health Intelligence Platform")

st.markdown(
    """
    Upload healthcare datasets, analyze patient trends,
    predict diabetes risk, and generate actionable insights.
    """
)

uploaded_file = st.file_uploader(
    "Upload Patient Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.subheader("Health Analytics")

    fig, ax = plt.subplots()

    df["Glucose"].hist(ax=ax)

    ax.set_title("Glucose Distribution")

    st.pyplot(fig)
    st.dataframe(df.head())
    st.subheader("Feature Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Patients",
            len(df)
        )

    with col2:
        st.metric(
            "Average Age",
            round(df["Age"].mean(), 1)
        )

    with col3:
        st.metric(
            "Average BMI",
            round(df["BMI"].mean(), 1)
        )

    with col4:
        st.metric(
            "Diabetes Cases",
            int(df["Outcome"].sum())
        )
    with st.expander("Dataset Statistics"):
        st.write(df.describe())
    if "Outcome" in df.columns:
        st.subheader("Diabetes Outcome Distribution")

        outcome_counts = df["Outcome"].value_counts()

        fig, ax = plt.subplots()

        ax.pie(
            outcome_counts,
            labels=["No Diabetes", "Diabetes"],
            autopct="%1.1f%%"
        )

        st.pyplot(fig)

    numeric_cols = df.select_dtypes(include=np.number).columns

    feature = st.selectbox(
        "Choose Feature",
        numeric_cols
    )

    fig, ax = plt.subplots()

    df[feature].hist(ax=ax)

    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Number of Patients")

    st.pyplot(fig)

    if "Outcome" in df.columns:

        model, accuracy, feature_names, cm,results= train_model(df)

        st.success(
            f"Model Accuracy: {accuracy * 100:.2f}%"
        )
        st.subheader("Model Comparison")
        for model_name,score in results.items():
            st.metric(
                model_name,
                f"{score*100:.2f}"
            )

        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        st.pyplot(fig)

        st.subheader("Feature Importance")

        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })

        feature_importance = feature_importance.sort_values(
            by="Importance",
            ascending=False
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(
            feature_importance["Feature"],
            feature_importance["Importance"]
        )

        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")

        st.pyplot(fig)
        st.subheader("Patient Risk Prediction")

        values = {}

        for col in df.columns:

            if col != "Outcome":

                values[col] = st.number_input(
                    col,
                    float(df[col].min()),
                    float(df[col].max()),
                    float(df[col].mean())
                )

        if st.button("Predict"):

            user = pd.DataFrame([values])

            prediction = model.predict(user)[0]

            probability = model.predict_proba(user)[0][1]

            risk_percent = probability * 100

            st.metric(
                "Risk Score",
                f"{risk_percent:.1f}%"
            )
            st.progress(float(probability))

            if risk_percent < 30:
                st.success("🟢 Low Risk")

            elif risk_percent < 70:
                st.warning("🟡 Moderate Risk")

            else:
                st.error("🔴 High Risk")

                st.subheader("Health Insights")

                if risk_percent < 30:

                    st.info(
                        """
                        Your health indicators suggest a relatively low risk.

                        Recommendations:
                        • Maintain a balanced diet
                        • Exercise regularly
                        • Continue routine health checkups
                        """
                    )

                elif risk_percent < 70:

                    st.warning(
                        """
                        Some health indicators require attention.

                        Recommendations:
                        • Reduce sugar intake
                        • Increase physical activity
                        • Monitor glucose levels
                        • Schedule a medical consultation
                        """
                    )

                else:

                    st.error(
                        """
                        Your risk profile appears elevated.

                        Recommendations:
                        • Consult a healthcare professional
                        • Monitor blood glucose regularly
                        • Consider lifestyle modifications
                        • Seek medical evaluation promptly
                        """
                    )

            st.subheader("Why This Prediction?")

            if values["Glucose"] > 140:
                st.warning(
                    "High glucose level increases diabetes risk."
                )

            if values["BMI"] > 30:
                st.warning(
                    "BMI is above the healthy range."
                )

            if values["BloodPressure"] > 90:
                st.warning(
                    "Blood pressure is elevated."
                )

            if values["Age"] > 50:
                st.warning(
                    "Age is a contributing risk factor."
                )
                st.subheader("Recommended Actions")

                if risk_percent > 70:

                    st.error(
                        """
                        • Schedule medical consultation

                        • Monitor glucose levels daily

                        • Review diet and exercise habits

                        • Seek professional healthcare advice
                        """
                    )

                elif risk_percent > 30:

                    st.warning(
                        """
                        • Increase physical activity

                        • Reduce sugar consumption

                        • Monitor health indicators regularly

                        • Consider preventive health checkups
                        """
                    )

                else:

                    st.success(
                        """
                        • Maintain healthy lifestyle

                        • Continue regular exercise

                        • Keep up routine health screenings
                        """
                    )
            st.download_button(
                label="Download Dataset",
                data=df.to_csv(index=False),
                file_name="health_data.csv",
                mime="text/csv"
            )