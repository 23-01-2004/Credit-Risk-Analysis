# tab9_deposit_growth.py

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_deposit_growth_analysis(data1):
    st.subheader("🏦 Deposit Growth Analysis (Regression)")

    # Features & Target
    X = data1[['Estimated Income', 'Age', 'Customer Tenure']]
    y = data1['Bank Deposits']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train regression model
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Predictions
    y_pred = reg.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.write("**Regression Coefficients:**", dict(zip(X.columns, reg.coef_)))
    st.write("**Intercept:**", reg.intercept_)
    st.write("**R² Score:**", round(r2, 3))
    st.write("**MSE:**", round(mse, 2))

    # --- Plots ---
    st.markdown("### 📊 Plots with Insights")

    # 1. Actual vs Predicted
    fig1, ax1 = plt.subplots(figsize=(15,4))
    ax1.scatter(y_test, y_pred, alpha=0.6, color="teal")
    ax1.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'k--', lw=2)
    ax1.set_xlabel("Actual Deposits")
    ax1.set_ylabel("Predicted Deposits")
    ax1.set_title("Actual vs Predicted Deposits")
    st.pyplot(fig1)

    error_mean = np.mean(np.abs(y_test - y_pred))
    st.markdown(f"**Insight:** The model explains deposit variation with an R² of {round(r2,3)}. "
                f"On average, predictions deviate from actual values by about {round(error_mean,2)} units. "
                f"Points close to the diagonal indicate good predictions, while large deviations highlight weaker fit.")

    # 2. Residuals plot
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots(figsize=(15,4))
    sns.histplot(residuals, kde=True, ax=ax2, color="orange")
    ax2.set_title("Residual Distribution")
    ax2.set_xlabel("Residuals")
    st.pyplot(fig2)

    skewness = residuals.skew()
    if abs(skewness) < 0.5:
        skew_text = "fairly symmetric, suggesting errors are balanced."
    elif skewness > 0.5:
        skew_text = "positively skewed, meaning the model underestimates deposits for some customers."
    else:
        skew_text = "negatively skewed, meaning the model overestimates deposits for some customers."

    st.markdown(f"**Insight:** The residuals center around {round(residuals.mean(),2)}. "
                f"The distribution is {skew_text}")

    # 3. Feature vs Deposits Scatter with correlations
    st.markdown("#### Feature Relationships")
    for feature in X.columns:
        fig, ax = plt.subplots(figsize=(15,4))
        sns.scatterplot(x=data1[feature], y=data1['Bank Deposits'], ax=ax, alpha=0.6)
        sns.regplot(x=data1[feature], y=data1['Bank Deposits'], ax=ax, scatter=False, color="red")
        ax.set_title(f"{feature} vs Bank Deposits")
        st.pyplot(fig)

        corr = data1[feature].corr(data1['Bank Deposits'])
        slope = reg.coef_[list(X.columns).index(feature)]

        if abs(corr) > 0.6:
            strength = "strong"
        elif abs(corr) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"

        direction = "positive" if slope > 0 else "negative"

        st.markdown(f"**Insight:** {feature} has a {strength} {direction} relationship with deposits "
                    f"(correlation = {round(corr,2)}). This means that as {feature} "
                    f"{'increases' if slope>0 else 'decreases'}, deposits tend to "
                    f"{'rise' if slope>0 else 'fall'} on average.")
