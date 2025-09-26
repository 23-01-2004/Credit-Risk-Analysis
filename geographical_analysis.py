import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def avg_deposits_by_geo(df):
    """Bar plot + dynamic insights: Average Bank Deposits by Nationality and Loyalty Classification"""
    if {'Nationality', 'Bank Deposits', 'Loyalty Classification'}.issubset(df.columns):
        st.subheader("Geographical Analysis: Average Deposits by Nationality & Loyalty Tier")
        
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(6,3))  # compact for Streamlit
        sns.barplot(
            x='Nationality',
            y='Bank Deposits',
            hue='Loyalty Classification',
            data=df,
            estimator='mean',
            ci=None,
            palette='Set1',
            ax=ax
        )
        ax.set_title("Average Deposits by Nationality & Loyalty Tier", fontsize=10, fontweight='bold')
        ax.set_xlabel("Nationality", fontsize=8)
        ax.set_ylabel("Average Bank Deposits", fontsize=8)
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(title='Loyalty Tier', loc='upper right', fontsize=7, title_fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # --- Dynamic Insights ---
        avg_df = df.groupby(['Nationality', 'Loyalty Classification'])['Bank Deposits'].mean().reset_index()
        top_combo = avg_df.loc[avg_df['Bank Deposits'].idxmax()]
        low_combo = avg_df.loc[avg_df['Bank Deposits'].idxmin()]

        # Overall average deposits
        overall_avg = df['Bank Deposits'].mean()

        # Gap between loyalty tiers by nationality
        gap_df = avg_df.groupby("Nationality")['Bank Deposits'].agg(lambda x: x.max() - x.min()).reset_index()
        max_gap = gap_df.loc[gap_df['Bank Deposits'].idxmax()]
        min_gap = gap_df.loc[gap_df['Bank Deposits'].idxmin()]

        st.markdown(
            f"""
            <div class="insight-card">
                <p><b> Insights:</b></p>
                <ul>
                    <li> <b>Highest Avg Deposit:</b> {top_combo['Nationality']} - {top_combo['Loyalty Classification']} with {top_combo['Bank Deposits']:.2f}</li>
                    <li> <b>Lowest Avg Deposit:</b> {low_combo['Nationality']} - {low_combo['Loyalty Classification']} with {low_combo['Bank Deposits']:.2f}</li>
                    <li> <b>Overall Avg Deposit:</b> {overall_avg:.2f}</li>
                    <li> <b>Widest Gap:</b> {max_gap['Nationality']} shows the largest difference between loyalty tiers ({max_gap['Bank Deposits']:.2f})</li>
                    <li> <b>Most Consistent:</b> {min_gap['Nationality']} has the smallest gap between loyalty tiers ({min_gap['Bank Deposits']:.2f})</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


# def create_geographical_dashboard(df):
#     """Geographical dashboard with plots and insights"""
#     avg_deposits_by_geo(df)
