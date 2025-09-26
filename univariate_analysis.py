import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from scipy.stats import skew, kurtosis

# -------------------------------
# Demographics Plots with Deep Insights
# -------------------------------
def demographics_plots(df):
    st.subheader("Demographics")
    
    # Age Distribution
    if 'Age' in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df['Age'], kde=True, bins=30, color='skyblue', ax=ax)
        ax.set_title("Age Distribution", fontsize=11, fontweight='bold')
        ax.set_xlabel("Age", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Dynamic insights
        mean_age = df['Age'].mean()
        median_age = df['Age'].median()
        min_age = df['Age'].min()
        max_age = df['Age'].max()
        age_skew = skew(df['Age'])
        age_kurt = kurtosis(df['Age'])
        st.info(f"📌 **Insight:** Customers are aged {min_age}-{max_age}. "
                f"Mean: {mean_age:.1f}, Median: {median_age:.1f}. "
                f"Distribution is {'right-skewed' if age_skew>0 else 'left-skewed' if age_skew<0 else 'symmetric'} "
                f"with kurtosis {age_kurt:.2f}, indicating {'heavy tails' if age_kurt>3 else 'light tails'}.")

    # Nationality Distribution
    if 'Nationality' in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        nationality_counts = df['Nationality'].value_counts()
        if len(nationality_counts) > 8:
            nationality_counts = nationality_counts.head(8)
        sns.countplot(data=df[df['Nationality'].isin(nationality_counts.index)],
                      x='Nationality', order=nationality_counts.index, ax=ax, palette='viridis')
        ax.set_title("Nationality Distribution", fontsize=11, fontweight='bold')
        ax.set_xlabel("Nationality", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        top_nat = nationality_counts.idxmax()
        top_count = nationality_counts.max()
        top_pct = top_count / len(df) * 100
        st.info(f"🌍 **Insight:** Top nationality: {top_nat} ({top_pct:.1f}% of dataset). "
                f"Remaining customers show diversity across {len(nationality_counts)-1} other nationalities. "
                f"Opportunities exist for targeted marketing to less-represented nationalities.")

    # Loyalty Classification
    if 'Loyalty Classification' in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        loyalty_counts = df['Loyalty Classification'].value_counts()
        colors = plt.cm.Set3(range(len(loyalty_counts)))
        bottom = 0
        for i, (category, count) in enumerate(loyalty_counts.items()):
            ax.bar(['Loyalty Distribution'], [count], bottom=bottom, 
                   color=colors[i], label=f'{category} ({count})')
            bottom += count
        ax.set_title("Loyalty Classification Distribution", fontsize=11, fontweight='bold')
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        dominant_loyalty = loyalty_counts.idxmax()
        loyalty_pct = loyalty_counts.max() / len(df) * 100
        low_loyalty_pct = 100 - loyalty_pct
        st.info(f"⭐ **Insight:** Dominant loyalty tier: {dominant_loyalty} ({loyalty_pct:.1f}%). "
                f"Other tiers constitute {low_loyalty_pct:.1f}% of customers. "
                f"Potential to convert mid-tier customers to higher loyalty through targeted campaigns.")

# -------------------------------
# Financials Plots with Deep Insights
# -------------------------------
def financials_plots(df):
    st.subheader("Financials")
    financial_cols = ['Estimated Income', 'Bank Deposits', 'Bank Loans']
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    titles = ['Income Distribution', 'Deposit Distribution', 'Loans Distribution']

    for col in financial_cols:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(df[col], kde=True, bins=30, color=colors[financial_cols.index(col)], ax=ax)
            ax.set_title(titles[financial_cols.index(col)], fontsize=11, fontweight='bold')
            ax.set_xlabel(col, fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='plain', axis='x')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Dynamic insights
            mean_val = df[col].mean()
            median_val = df[col].median()
            min_val = df[col].min()
            max_val = df[col].max()
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            skew_val = skew(df[col])
            kurt_val = kurtosis(df[col])
            outliers = ((df[col] < q25 - 1.5*(q75-q25)) | (df[col] > q75 + 1.5*(q75-q25))).sum()

            st.info(f"💰 **{col} Insight:** Mean={mean_val:,.0f}, Median={median_val:,.0f}, <br>"
                    f"Range={min_val:,.0f}-{max_val:,.0f}, 25th-75th percentile={q25:,.0f}-{q75:,.0f}. "
                    f"Skew={skew_val:.2f} ({'right' if skew_val>0 else 'left' if skew_val<0 else 'symmetric'}-skewed), "
                    f"Kurtosis={kurt_val:.2f}. Detected {outliers} potential outliers.")

# -------------------------------
# Categorical Variables with Insights
# -------------------------------
def categorical_plots(df):
    st.subheader("Categorical Variables")
    if 'Fee Structure' in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        fee_counts = df['Fee Structure'].value_counts()
        colors = plt.cm.Pastel1(range(len(fee_counts)))
        bottom = 0
        for i, (category, count) in enumerate(fee_counts.items()):
            ax.bar(['Fee Structure'], [count], bottom=bottom, color=colors[i], label=f'{category} ({count})')
            bottom += count
        ax.set_title("Fee Structure Distribution", fontsize=11, fontweight='bold')
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        dominant_fee = fee_counts.idxmax()
        fee_pct = fee_counts.max() / len(df) * 100
        low_fee_pct = 100 - fee_pct
        st.info(f"📊 **Insight:** Most customers are on **{dominant_fee}** ({fee_pct:.1f}%). "
                f"Other fee structures cover {low_fee_pct:.1f}% of customers. "
                f"Opportunities exist to upsell premium fee plans to low-tier customers.")

# -------------------------------
# Dashboard Creator
# -------------------------------
def create_dashboard(df):
    st.title("Comprehensive Customer Analytics Dashboard")
    st.markdown("### 🧾 Demographics Analysis")
    demographics_plots(df)

    st.markdown("### 💹 Financial Analysis")
    financials_plots(df)

    st.markdown("### 🗂 Categorical Analysis")
    categorical_plots(df)
