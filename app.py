import streamlit as st
import pandas as pd
from banking_analysis import load_and_preprocess, summarize_data
from outlier_detection import plot_boxplots_before_after
from feature_engineering import feature_engineering
from univariate_analysis import demographics_plots, financials_plots, categorical_plots, create_dashboard
from bivariate_analysis import create_bivariate_dashboard
from geographical_analysis import avg_deposits_by_geo
from customer_segmentation import clustering_dashboard
from credit_risk_modelling import run_model_comparison
from deposit_growth_analysis import run_deposit_growth_analysis

# Page configuration
st.set_page_config(
    page_title="Banking Data Analytics Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main app styling */
    .main {
        padding: 2rem 1rem;
    }
    
  
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        color: white !important;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
        color: white !important;
    }
    
    /* Card styling */
    .info-card {
        background: black;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: black;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* File uploader styling */
    .upload-section {
        background: grey;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #cbd5e1;
        text-align: center;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #3b82f6;
        background: #f1f5f9;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: lightgrey;
        padding: 0.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: black;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f1f5f9;
        border-color: #3b82f6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
        border-color: #3b82f6 !important;
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        color: white !important;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
        text-align: center;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
    }
    
    /* Section header styling */
    .section-header {
        background: linear-gradient(90deg, #64748b 0%, #94a3b8 100%);
        color: white !important;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Control panel styling */
    .control-panel {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏦 Banking Data Analytics Platform</h1>
    <p>Comprehensive analysis and insights for banking datasets</p>
</div>
""", unsafe_allow_html=True)

# File upload section
st.markdown("""
<div class="upload-section">
    <h3>📁 Upload Your Banking Dataset</h3>
    <p>Please upload a CSV file to begin your analysis journey</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload your banking dataset in CSV format"
)

if uploaded_file is not None:
    # Load and preprocess data
    with st.spinner('Loading and preprocessing data...'):
        df = load_and_preprocess(uploaded_file)
        summary = summarize_data(df)
    
    st.success(f"✅ Successfully loaded dataset with {summary['shape'][0]:,} rows and {summary['shape'][1]} columns")
    
    # Create tabs with enhanced styling
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Dataset Overview",
        "🎯 Outlier Detection", 
        "⚙️ Feature Engineering",
        "📈 Univariate Analysis",
        "🔍 Bivariate Analysis",
        "🌍 Geographical Insights",
        "👥 Customer Segmentation",
        "💳 Credit Risk Modeling",
        "💰 Deposit Growth Analysis"
    ])
    
    # Dataset Overview Tab
    with tab1:
        st.markdown('<div class="section-header">📊 Dataset Overview & Summary</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📋 Dataset Shape</h4>
                <p><strong>Rows:</strong> {summary['shape'][0]:,}</p>
                <p><strong>Columns:</strong> {summary['shape'][1]:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            null_count = sum(summary['nulls'].values())
            st.markdown(f"""
            <div class="metric-card">
                <h4>❌ Missing Values</h4>
                <p><strong>Total Nulls:</strong> {null_count:,}</p>
                <p><strong>Columns with Nulls:</strong> {sum(1 for v in summary['nulls'].values() if v > 0)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            numeric_cols = len(df.select_dtypes(include='number').columns)
            categorical_cols = len(df.select_dtypes(exclude='number').columns)
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔢 Column Types</h4>
                <p><strong>Numeric:</strong> {numeric_cols}</p>
                <p><strong>Categorical:</strong> {categorical_cols}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.subheader("❌ Null Values by Column")
            null_df = pd.DataFrame.from_dict(summary["nulls"], orient="index", columns=["Null Count"])
            null_df = null_df[null_df["Null Count"] > 0].sort_values("Null Count", ascending=False)
            if not null_df.empty:
                st.dataframe(null_df, use_container_width=True)
            else:
                st.success("🎉 No missing values found in the dataset!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.subheader("🔢 Unique Values by Column")
            unique_df = pd.DataFrame.from_dict(summary["unique_values"], orient="index", columns=["Unique Count"])
            unique_df = unique_df.sort_values("Unique Count", ascending=False).head(10)
            st.dataframe(unique_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📊 Statistical Summary")
        st.dataframe(summary["description"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Outlier Detection Tab
    with tab2:
        st.markdown('<div class="section-header">🎯 Outlier Detection & Treatment</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("🔧 Analysis Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            selected_cols = st.multiselect(
                "📊 Select columns for outlier treatment:",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,
                help="Choose which numeric columns to analyze for outliers"
            )
        
        with col2:
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                width = st.slider("📏 Plot Width", 2, 12, 8, help="Adjust plot width")
            with col2_2:
                height = st.slider("📐 Plot Height", 1, 4, 2, help="Adjust plot height")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if selected_cols:
            with st.spinner('Detecting and treating outliers...'):
                df = plot_boxplots_before_after(df, selected_cols, width, height)
            
            st.markdown("""
            <div class="success-message">
                ✅ Outlier detection and treatment completed successfully!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Please select at least one column for outlier analysis.")
    
    # Feature Engineering Tab
    with tab3:
      st.markdown('<div class="section-header">⚙️ Feature Engineering Pipeline</div>', unsafe_allow_html=True)
      
      # Apply feature engineering
      with st.spinner('Applying feature engineering transformations...'):
          df_fe, new_features, insights = feature_engineering(df)  # unpack all three values
      
      # Enhanced Dataset Preview
      st.markdown('<div class="info-card">', unsafe_allow_html=True)
      st.subheader(" Enhanced Dataset Preview")
      st.dataframe(df_fe.head(), use_container_width=True)
      st.markdown('</div>', unsafe_allow_html=True)
      
      # Newly Created Features
      if new_features:
          st.markdown('<div class="info-card">', unsafe_allow_html=True)
          st.subheader(" Newly Created Features")
          st.markdown("<ul>", unsafe_allow_html=True)
          for feature in new_features:
              st.markdown(f"<li><b>{feature}</b></li>", unsafe_allow_html=True)
          st.markdown("</ul>", unsafe_allow_html=True)
          st.markdown('</div>', unsafe_allow_html=True)

          # Dynamic Insights
          if insights:
              st.markdown('<div class="info-card">', unsafe_allow_html=True)
              st.subheader(" Dynamic Insights")
              for insight in insights:
                  st.markdown(f"- {insight}")
              st.markdown('</div>', unsafe_allow_html=True)
      else:
          st.info("ℹ️ No new features were created in this iteration.")
      
      # Success Message
      st.markdown("""
      <div class="success-message">
          ✅ Feature engineering pipeline executed successfully!
      </div>
      """, unsafe_allow_html=True)



    
   # Univariate Analysis Tab
    with tab4:
        st.markdown('<div class="section-header">📈 Univariate Analysis Dashboard</div>', unsafe_allow_html=True)
        
        # Combined dashboard handles headers, plots, and dynamic insights
        create_dashboard(df)

    
    # Bivariate Analysis Tab
    with tab5:
        st.markdown('<div class="section-header">🔍 Bivariate Relationship Analysis</div>', unsafe_allow_html=True)
        create_bivariate_dashboard(df)
    
    # Geographical Analysis Tab
    with tab6:
        st.markdown('<div class="section-header">🌍 Geographical Insights Dashboard</div>', unsafe_allow_html=True)
        avg_deposits_by_geo(df)
    
    # Customer Segmentation Tab
    with tab7:
        st.markdown('<div class="section-header">👥 Customer Segmentation Analysis</div>', unsafe_allow_html=True)
        
        with st.spinner('Applying advanced feature engineering for segmentation...'):
            df_feature_engineered, new_features, insights = feature_engineering(df)

        
        with st.spinner('Performing customer segmentation analysis...'):
            df_segmented = clustering_dashboard(df_feature_engineered)
    
    # Credit Risk Modeling Tab
    with tab8:
        st.markdown('<div class="section-header">💳 Credit Risk Modeling Suite</div>', unsafe_allow_html=True)
        
        with st.spinner('Running comprehensive model comparison...'):
            run_model_comparison(df_feature_engineered)
    
    # Deposit Growth Analysis Tab
    with tab9:
        st.markdown('<div class="section-header">💰 Deposit Growth Analysis</div>', unsafe_allow_html=True)
        
        with st.spinner('Analyzing deposit growth patterns...'):
            run_deposit_growth_analysis(df_feature_engineered)

else:
    # Instructions when no file is uploaded
    st.markdown("""
    <div class="info-card">
        <h3>🚀 Getting Started</h3>
        <p>Welcome to the Banking Data Analytics Platform! This comprehensive tool provides:</p>
        <ul>
            <li><strong>📊 Dataset Overview:</strong> Explore your data structure and summary statistics</li>
            <li><strong>🎯 Outlier Detection:</strong> Identify and treat anomalous data points</li>
            <li><strong>⚙️ Feature Engineering:</strong> Create new meaningful features automatically</li>
            <li><strong>📈 Univariate Analysis:</strong> Analyze individual variables in depth</li>
            <li><strong>🔍 Bivariate Analysis:</strong> Discover relationships between variables</li>
            <li><strong>🌍 Geographical Insights:</strong> Visualize geographic patterns in your data</li>
            <li><strong>👥 Customer Segmentation:</strong> Group customers based on behavior patterns</li>
            <li><strong>💳 Credit Risk Modeling:</strong> Build and compare predictive models</li>
            <li><strong>💰 Deposit Growth Analysis:</strong> Analyze trends in deposit behavior</li>
        </ul>
        <p><strong>To begin:</strong> Upload your banking dataset in CSV format using the file uploader above.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; font-size: 0.9rem; margin-top: 2rem;">
    <p>🏦 Banking Data Analytics Platform | Built with Streamlit & Python</p>
</div>
""", unsafe_allow_html=True)