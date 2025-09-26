import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

def parse_date(x):
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return pd.to_datetime(x, format=fmt)
        except:
            continue
    return pd.NaT


def feature_engineering(df):
    """
    Apply feature engineering and return:
        - df_copy: dataframe with new features
        - new_features: list of columns newly added
        - insights: list of dynamic insights generated from the data
    """
    df_copy = df.copy()
    original_cols = set(df_copy.columns)

    # Customer Tenure
    if 'Joined Bank' in df_copy.columns:
        df_copy['Joined Bank'] = df_copy['Joined Bank'].apply(parse_date)
        today = pd.to_datetime(datetime.today().date())
        df_copy['Customer Tenure'] = ((today - df_copy['Joined Bank']).dt.days / 365).round(1)

    # Total Relationship Balance
    components = ['Checking Accounts', 'Saving Accounts', 'Foreign Currency Account']
    for col in components:
        if col not in df_copy.columns:
            df_copy[col] = 0
    df_copy['Total Relationship Balance'] = df_copy[components].sum(axis=1)

    # Debt-to-Income Ratio
    if {'Bank Loans','Credit Card Balance','Estimated Income'}.issubset(df_copy.columns):
        df_copy['Debt-to-Income Ratio'] = (df_copy['Bank Loans'] + df_copy['Credit Card Balance']) / df_copy['Estimated Income']
        df_copy['Debt-to-Income Ratio'] = df_copy['Debt-to-Income Ratio'].replace([np.inf, -np.inf], np.nan)

    # Deposit-to-Loan Ratio
    if {'Bank Deposits', 'Bank Loans'}.issubset(df_copy.columns):
        df_copy['Deposit-to-Loan Ratio'] = df_copy['Bank Deposits']/ df_copy['Bank Loans'].replace(0,np.nan)

    # Wealth Indicator
    if {'Superannuation Savings','Properties Owned'}.issubset(df_copy.columns):
        assumed_property_value = 500000
        df_copy['Wealth Indicator'] = (
            df_copy['Total Relationship Balance'] +
            df_copy['Superannuation Savings'] +
            df_copy['Properties Owned'] * assumed_property_value
        )

    # Product Concentration
    product_cols = ['Checking Accounts', 'Saving Accounts', 'Foreign Currency Account', 
                    'Credit Card Balance', 'Bank Loans', 'Bank Deposits']
    present_products = [col for col in product_cols if col in df_copy.columns]
    df_copy['Product Concentration'] = (df_copy[present_products] > 0).sum(axis=1)

    # Age x Total Relationship Balance
    if {'Age', 'Total Relationship Balance'}.issubset(df_copy.columns):
        df_copy['Age_x_Balance'] = df_copy['Age'] * df_copy['Total Relationship Balance']

    # Age Groups
    if 'Age' in df_copy.columns:
        bins = [0,25,40,60,75,np.inf]
        labels = ['Gen Z', 'Millenial', 'Gen X', 'Baby Boomer', 'Silent Generation']
        df_copy['Age Group'] = pd.cut(df_copy['Age'], bins=bins, labels=labels, right=True)

    # Income Groups
    if 'Estimated Income' in df_copy.columns:
        bins = [0, 200000, 300000, 400000 , np.inf]
        labels = ['Low Net Worth', 'Medium Net Worth', 'High Net Worth', 'Premium Customers']
        df_copy['Income Group'] = pd.cut(df_copy['Estimated Income'], bins=bins, labels=labels, right=True)

    # Determine which features were newly added
    new_features = list(set(df_copy.columns) - original_cols)

    # Generate dynamic insights
    insights = []
    if 'Customer Tenure' in new_features:
        avg_tenure = df_copy['Customer Tenure'].mean()
        insights.append(f" The average **Customer Tenure** is **{avg_tenure:.1f} years**.")
    
    if 'Debt-to-Income Ratio' in new_features:
        risky = (df_copy['Debt-to-Income Ratio'] > 0.5).mean() * 100
        insights.append(f" About **{risky:.1f}%** of customers have a **Debt-to-Income Ratio above 0.5**, indicating higher financial risk.")
    
    if 'Deposit-to-Loan Ratio' in new_features:
        avg_ratio = df_copy['Deposit-to-Loan Ratio'].mean()
        insights.append(f" The average **Deposit-to-Loan Ratio** is **{avg_ratio:.2f}**, showing overall liquidity strength.")
    
    if 'Wealth Indicator' in new_features:
        top_wealth = df_copy['Wealth Indicator'].max()
        insights.append(f" The wealthiest customer has a **Wealth Indicator** of **{top_wealth:,.0f}**.")
    
    if 'Product Concentration' in new_features:
        avg_products = df_copy['Product Concentration'].mean()
        insights.append(f" On average, customers hold **{avg_products:.1f} products** with the bank.")
    
    if 'Age Group' in new_features:
        dominant_age = df_copy['Age Group'].mode()[0]
        insights.append(f" The most common **Age Group** is **{dominant_age}**.")
    
    if 'Income Group' in new_features:
        dominant_income = df_copy['Income Group'].mode()[0]
        insights.append(f" The majority of customers fall under the **{dominant_income}** segment.")
    
    return df_copy, new_features, insights