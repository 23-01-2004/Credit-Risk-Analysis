import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Set style for beautiful plots
plt.style.use('default')
sns.set_palette("husl")

# -------------------------------
# Individual Plot Functions
# -------------------------------

def violin_plot(df):
    """Enhanced violin plot with dynamic insights"""
    if {'Loyalty Classification', 'Estimated Income'}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Custom color palette
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        sns.violinplot(
            x='Loyalty Classification', 
            y='Estimated Income', 
            data=df, 
            inner='quart',
            palette=colors[:df['Loyalty Classification'].nunique()],
            ax=ax,
            alpha=0.8
        )
        
        ax.set_title("💰 Income Distribution by Loyalty Tier", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Loyalty Classification", fontsize=12, fontweight='bold')
        ax.set_ylabel("Estimated Income ($)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#F8F9FA')
        plt.xticks(rotation=45, fontsize=10, ha='right')
        plt.yticks(fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        for spine in ax.spines.values():
            spine.set_edgecolor('#DDDDDD')
            spine.set_linewidth(1)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Dynamic insight: median income per loyalty tier
        medians = df.groupby('Loyalty Classification')['Estimated Income'].median()
        highest_tier = medians.idxmax()
        lowest_tier = medians.idxmin()
        st.info(f"📊 **Insight:** Median income is highest for **{highest_tier}** and lowest for **{lowest_tier}**. "
                f"The width of each violin shows variability; wider shapes indicate more income spread within that tier.")

# # -------------------------------
# def scatter_tenure_deposits(df):
#     """Scatter plot for tenure vs deposits with dynamic insights"""
#     if {'Customer Tenure', 'Bank Deposits', 'Estimated Income'}.issubset(df.columns):
#         fig, ax = plt.subplots(figsize=(10, 6))
#         scatter = ax.scatter(
#             df['Customer Tenure'], 
#             df['Bank Deposits'], 
#             c=df['Estimated Income'], 
#             cmap='viridis', 
#             alpha=0.7,
#             s=50, edgecolors='white', linewidth=0.5
#         )
        
#         ax.set_xlabel("👥 Customer Tenure (years)", fontsize=12, fontweight='bold')
#         ax.set_ylabel("💳 Bank Deposits ($)", fontsize=12, fontweight='bold')
#         ax.set_title("🔍 Customer Tenure vs Bank Deposits", fontsize=16, fontweight='bold', pad=20)
#         ax.grid(True, alpha=0.3, linestyle='--')
#         ax.set_facecolor('#F8F9FA')
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
#         cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
#         cbar.set_label('💰 Estimated Income ($)', fontsize=10, fontweight='bold')
#         cbar.ax.tick_params(labelsize=9)
#         cbar.formatter = plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
#         cbar.update_ticks()
#         for spine in ax.spines.values():
#             spine.set_edgecolor('#DDDDDD')
#             spine.set_linewidth(1)
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close(fig)
        
#         # Dynamic insight: correlation
#         corr = df['Customer Tenure'].corr(df['Bank Deposits'])
#         st.metric("📈 Correlation Coefficient", f"{corr:.3f}")
#         direction = "increases" if corr > 0 else "decreases"
#         st.info(f"📊 **Insight:** There is a {abs(corr):.2f} correlation between tenure and deposits, "
#                 f"indicating that as customer tenure {direction}, bank deposits generally {direction} as well.")

# # -------------------------------
# def scatter_risk_dti(df):
#     """Scatter plot for risk weighting vs debt-to-income ratio with dynamic insights"""
#     if {'Risk Weighting', 'Debt-to-Income Ratio', 'Estimated Income'}.issubset(df.columns):
#         fig, ax = plt.subplots(figsize=(10, 6))
#         scatter = ax.scatter(
#             df['Risk Weighting'], 
#             df['Debt-to-Income Ratio'], 
#             c=df['Estimated Income'], 
#             cmap='plasma', 
#             alpha=0.7,
#             s=50, edgecolors='white', linewidth=0.5
#         )
        
#         ax.set_xlabel("⚠️ Risk Weighting", fontsize=12, fontweight='bold')
#         ax.set_ylabel("📊 Debt-to-Income Ratio", fontsize=12, fontweight='bold')
#         ax.set_title("🎯 Risk Assessment Analysis", fontsize=16, fontweight='bold', pad=20)
#         ax.grid(True, alpha=0.3, linestyle='--')
#         ax.set_facecolor('#F8F9FA')
        
#         cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
#         cbar.set_label('💰 Estimated Income ($)', fontsize=10, fontweight='bold')
#         cbar.ax.tick_params(labelsize=9)
#         cbar.formatter = plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
#         cbar.update_ticks()
        
#         for spine in ax.spines.values():
#             spine.set_edgecolor('#DDDDDD')
#             spine.set_linewidth(1)
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close(fig)
        
#         corr = df['Risk Weighting'].corr(df['Debt-to-Income Ratio'])
#         st.metric("📈 Risk-DTI Correlation", f"{corr:.3f}")
#         st.info(f"📊 **Insight:** The correlation between risk weighting and debt-to-income ratio is {corr:.2f}, "
#                 f"indicating that higher risk scores generally correspond to {'higher' if corr>0 else 'lower'} DTI ratios.")

# -------------------------------
def correlation_heatmap(df):
    """Correlation heatmap with dynamic insights"""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        colors = ['#FF6B6B', '#FFFFFF', '#4ECDC4']
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, fmt=".3f", cmap=cmap,
            center=0, square=True, ax=ax, cbar_kws={"shrink":0.8, "label":"Correlation Coefficient"},
            annot_kws={"size":10, "weight":"bold"}
        )
        ax.set_title("🔗 Feature Correlation Matrix", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Dynamic insight: top correlations
        corr_pairs = corr_matrix.abs().unstack().sort_values(kind="quicksort", ascending=False)
        corr_pairs = corr_pairs[corr_pairs < 1]  # remove diagonal
        top = corr_pairs.head(3)
        for i, (pair, value) in enumerate(top.items(), 1):
            f1, f2 = pair
            direction = "positive" if corr_matrix.loc[f1,f2] > 0 else "negative"
            st.info(f"🔹 **Top {i} correlation:** {f1} ↔ {f2} = {corr_matrix.loc[f1,f2]:.2f} ({direction})")

# -------------------------------
# Dashboard Creator
# -------------------------------

def create_bivariate_dashboard(df):
    """Bivariate Analysis Dashboard with dynamic insights"""
    tab_violin,  tab_corr = st.tabs([
        "🎻 Income by Loyalty", 
        # "📈 Tenure vs Deposits", 
        # "⚖️ Risk Analysis", 
        "🔗 Correlations"
    ])
    
    with tab_violin:
        st.markdown("### 🎻 Income Distribution by Loyalty Tier")
        violin_plot(df)
    
    # with tab_scatter_td:
    #     st.markdown("### 📈 Customer Tenure vs Bank Deposits")
    #     scatter_tenure_deposits(df)
    
    # with tab_scatter_risk:
    #     st.markdown("### ⚖️ Risk Weighting vs Debt-to-Income Analysis")
    #     scatter_risk_dti(df)
    
    with tab_corr:
        st.markdown("### 🔗 Feature Correlation Analysis")
        correlation_heatmap(df)

# -------------------------------
# Optional CSS for better styling
# -------------------------------
def apply_custom_css():
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; padding: 0 20px; background-color: #f0f2f6; border-radius: 10px; color: #262730; font-weight: bold; }
    .stTabs [aria-selected="true"] { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; }
    .stMetric { background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 5px solid #667eea; }
    </style>
    """, unsafe_allow_html=True)
