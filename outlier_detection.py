import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def cap_outliers(series):
    """
    Cap outliers using IQR method.
    """
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return np.where(series > upper, upper,
           np.where(series < lower, lower, series))


def plot_boxplots_before_after(df, cols, width=6, height=1):
    """
    Plot before and after boxplots side by side for each column.
    Returns modified DataFrame.
    """
    df_copy = df.copy()

    for col in cols:
        if col in df_copy.columns:
            fig, axes = plt.subplots(1, 2, figsize=(width, height))

            # Before
            sns.boxplot(x=df[col], ax=axes[0])
            axes[0].set_title(f'Before - {col}', fontsize=9)

            # Apply outlier capping
            df_copy[col] = cap_outliers(df_copy[col])

            # After
            sns.boxplot(x=df_copy[col], ax=axes[1])
            axes[1].set_title(f'After - {col}', fontsize=9)

            st.pyplot(fig, clear_figure=True)

    return df_copy
