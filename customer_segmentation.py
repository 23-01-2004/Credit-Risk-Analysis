import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
import pandas as pd

def clustering_dashboard(df):
    """
    Customer Segmentation Dashboard with dynamic insights
    df: feature-engineered dataframe with columns:
        'Customer Tenure', 'Product Concentration', 
        'Total Relationship Balance', 'Estimated Income'
    Returns:
        df with cluster labels
    """

    # Required columns
    required_cols = ['Customer Tenure', 'Product Concentration', 
                     'Total Relationship Balance', 'Estimated Income']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns for clustering: {missing_cols}")
        return df

    rfm_features = df[required_cols]

    # Standardize features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(rfm_scaled)
    df['KMeans_Segment'] = kmeans_labels

    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=4, random_state=42)
    gmm_labels = gmm.fit_predict(rfm_scaled)
    df['GMM_Segment'] = gmm_labels

    # K-Medoids (PAM)
    kmedoids = KMedoids(n_clusters=4, random_state=42, method='pam')
    kmedoids_labels = kmedoids.fit_predict(rfm_scaled)
    df['PAM_Segment'] = kmedoids_labels

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(rfm_scaled)

    # Sub-tabs for each clustering technique
    tab_kmeans, tab_gmm, tab_pam = st.tabs(["KMeans", "Gaussian Mixture Model (GMM)", "Partition Around Medoids (PAM)"])

    # --- Helper: Generate Insights ---
    def clustering_insights(df, label_col, method_name):
        st.markdown(f"### 🔍 Insights: {method_name}")
        cluster_summary = df.groupby(label_col)[required_cols].mean()
        counts = df[label_col].value_counts().sort_index()

        # Top segments
        top_income = cluster_summary['Estimated Income'].idxmax()
        top_balance = cluster_summary['Total Relationship Balance'].idxmax()

        st.markdown(
            f"""
            <div class="insight-card">
                <ul>
                    <li><b>Number of Segments:</b> {counts.shape[0]}</li>
                    <li><b>Largest Segment:</b> Cluster {counts.idxmax()} with {counts.max()} customers</li>
                    <li><b>Smallest Segment:</b> Cluster {counts.idxmin()} with {counts.min()} customers</li>
                    <li><b>Wealthiest Segment:</b> Cluster {top_income} (highest avg income)</li>
                    <li><b>Strongest Relationship:</b> Cluster {top_balance} (highest avg relationship balance)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Optional: show summary dataframe
        st.dataframe(cluster_summary.style.highlight_max(color="lightgreen", axis=0))

    # --- KMeans Plot + Insights ---
    with tab_kmeans:
        fig, ax = plt.subplots(figsize=(5,3))
        scatter = ax.scatter(pca_data[:,0], pca_data[:,1], c=kmeans_labels, cmap='viridis', alpha=0.7)
        ax.set_title("KMeans Clustering", fontsize=10, fontweight='bold')
        ax.set_xlabel("PCA Component 1", fontsize=8)
        ax.set_ylabel("PCA Component 2", fontsize=8)
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cluster", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        st.pyplot(fig)
        plt.close()

        clustering_insights(df, 'KMeans_Segment', "KMeans")

    # --- GMM Plot + Insights ---
    with tab_gmm:
        fig, ax = plt.subplots(figsize=(5,3))
        scatter = ax.scatter(pca_data[:,0], pca_data[:,1], c=gmm_labels, cmap='plasma', alpha=0.7)
        ax.set_title("Gaussian Mixture Model (GMM)", fontsize=10, fontweight='bold')
        ax.set_xlabel("PCA Component 1", fontsize=8)
        ax.set_ylabel("PCA Component 2", fontsize=8)
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cluster", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        st.pyplot(fig)
        plt.close()

        clustering_insights(df, 'GMM_Segment', "Gaussian Mixture Model (GMM)")

    # --- PAM Plot + Insights ---
    with tab_pam:
        fig, ax = plt.subplots(figsize=(5,3))
        scatter = ax.scatter(pca_data[:,0], pca_data[:,1], c=kmedoids_labels, cmap='inferno', alpha=0.7)
        ax.set_title("Partition Around Medoids (PAM)", fontsize=10, fontweight='bold')
        ax.set_xlabel("PCA Component 1", fontsize=8)
        ax.set_ylabel("PCA Component 2", fontsize=8)
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cluster", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        st.pyplot(fig)
        plt.close()

        clustering_insights(df, 'PAM_Segment', "Partition Around Medoids (PAM)")

    return df
