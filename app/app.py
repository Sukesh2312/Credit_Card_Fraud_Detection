import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    inject_custom_css,
    load_dataset,
    load_dataset_sample,
    load_results,
    load_model,
    get_cached_reference_stats,
    create_class_distribution_pie,
    create_grouped_class_comparison,
    create_amount_distribution_chart,
    create_correlation_heatmap,
    create_model_comparison_bar,
    get_feature_importance_df,
    create_feature_importance_chart,
    create_probability_histogram
)
from predictor import (
    inspect_uploaded_file,
    evaluate_automatic_selection,
    run_fraud_prediction
)

# Page Config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Styling
inject_custom_css()

# Load Results & Summaries
comp_df, cv_df, dist_df = load_results()

# Sidebar Navigation
st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <h2 style='color: #60a5fa; margin-bottom: 0;'>💳 FraudGuard AI</h2>
        <p style='color: #94a3b8; font-size: 0.85rem;'>Credit Card Fraud Detection System</p>
    </div>
""", unsafe_allow_html=True)

nav_page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Dataset Analysis", "Model Performance", "Fraud Prediction", "Feature Importance", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Project Info**\n\nAutomatic Dataset & Model Selection based on schema compatibility, statistical distribution similarity, and historical validation performance.")

# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if nav_page == "Overview":
    st.markdown("<h1>Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: #64748b;'>2013 vs 2023 Dataset Analysis & Machine Learning</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Dynamically compute KPI metrics from datasets
    df_13_sample = load_dataset_sample("2013", sample_size=300000)
    df_23_sample = load_dataset_sample("2023", sample_size=600000)
    
    tot_13 = len(df_13_sample) if not df_13_sample.empty else 284807
    tot_23 = len(df_23_sample) if not df_23_sample.empty else 568630
    
    fraud_rate_13 = (df_13_sample["Class"].mean() * 100) if not df_13_sample.empty else 0.1727
    fraud_rate_23 = (df_23_sample["Class"].mean() * 100) if not df_23_sample.empty else 50.00
    
    # 4 KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Total 2013 Transactions</div>
                <div class='kpi-value'>{tot_13:,}</div>
                <div class='kpi-subtitle'>Classic Kaggle Dataset</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Total 2023 Transactions</div>
                <div class='kpi-value'>{tot_23:,}</div>
                <div class='kpi-subtitle'>Modern Kaggle Dataset</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>2013 Fraud Rate</div>
                <div class='kpi-value kpi-value-fraud'>{fraud_rate_13:.2f}%</div>
                <div class='kpi-subtitle'>492 Fraud / {tot_13:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>2023 Fraud Rate</div>
                <div class='kpi-value kpi-value-safe'>{fraud_rate_23:.2f}%</div>
                <div class='kpi-subtitle'>284,315 Fraud / {tot_23:,}</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Side-by-side comparison charts
    c1, c2 = st.columns(2)
    with c1:
        fig1 = create_class_distribution_pie(df_13_sample, "2013")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = create_class_distribution_pie(df_23_sample, "2023")
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Grouped Bar Chart
    if not dist_df.empty:
        fig_dist = create_grouped_class_comparison(dist_df)
        st.plotly_chart(fig_dist, use_container_width=True)
        
    # Automatic Dataset Comparison Insight Card
    st.markdown("<br>", unsafe_allow_html=True)
    diff_rate = fraud_rate_23 - fraud_rate_13
    comparison_word = "higher" if diff_rate > 0 else "lower"
    st.markdown(f"""
        <div class='insight-card'>
            📊 <strong>Dataset Comparison Insight:</strong><br>
            Dataset comparison shows that the 2023 dataset has a significantly <strong>{comparison_word} fraud proportion ({fraud_rate_23:.2f}%)</strong> 
            than the 2013 dataset ({fraud_rate_13:.2f}%). The 2013 dataset represents real-world extreme class imbalance (0.17% fraud), 
            whereas the 2023 dataset is a balanced synthetic benchmark dataset (50% fraud).
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 2: DATASET ANALYSIS
# ==========================================
elif nav_page == "Dataset Analysis":
    st.markdown("<h1>Dataset Analysis & Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    ds_choice = st.radio("Select Dataset", ["2013 Dataset", "2023 Dataset"], horizontal=True)
    year = "2013" if "2013" in ds_choice else "2023"
    
    df_selected = load_dataset(year)
    
    if df_selected.empty:
        st.warning("Dataset unavailable.")
    else:
        tot_rows = len(df_selected)
        tot_cols = len(df_selected.columns)
        missing_vals = int(df_selected.isna().sum().sum())
        duplicates = int(df_selected.duplicated().sum())
        
        fraud_cnt = int((df_selected["Class"] == 1).sum()) if "Class" in df_selected.columns else 0
        legit_cnt = int((df_selected["Class"] == 0).sum()) if "Class" in df_selected.columns else 0
        fraud_pct = (fraud_cnt / tot_rows * 100) if tot_rows > 0 else 0.0
        
        # KPI Cards for Dataset Analysis
        k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
        k1.metric("Rows", f"{tot_rows:,}")
        k2.metric("Columns", tot_cols)
        k3.metric("Missing", missing_vals)
        k4.metric("Duplicates", f"{duplicates:,}")
        k5.metric("Legitimate", f"{legit_cnt:,}")
        k6.metric("Fraud", f"{fraud_cnt:,}")
        k7.metric("Fraud %", f"{fraud_pct:.2f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Dataset Preview (First 10 Rows)")
        st.dataframe(df_selected.head(10), use_container_width=True)
        
        with st.expander("Show Dataset Information"):
            info_df = pd.DataFrame({
                "Column Name": df_selected.columns,
                "Data Type": [str(t) for t in df_selected.dtypes],
                "Missing Values": df_selected.isna().sum().values,
                "Unique Values": [df_selected[c].nunique() for c in df_selected.columns]
            })
            st.write(f"**Total Duplicate Rows:** {duplicates:,}")
            st.dataframe(info_df, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Transaction Amount Distribution")
        fig_amt = create_amount_distribution_chart(df_selected, year)
        st.plotly_chart(fig_amt, use_container_width=True)
        
        st.markdown("""
            *Explanation:* The log-scaled amount distribution shows transaction amounts across legitimate vs fraudulent transactions. 
            Fraudulent transactions in real-world credit datasets often cluster around specific transaction size ranges.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Correlation Heatmap")
        fig_corr = create_correlation_heatmap(df_selected, year)
        st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# PAGE 3: MODEL PERFORMANCE
# ==========================================
elif nav_page == "Model Performance":
    st.markdown("<h1>Model Performance & Evaluation</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if comp_df.empty:
        st.error("Model comparison results file (`results/model_comparison.csv`) missing.")
    else:
        ds_sel = st.selectbox("Select Dataset", ["2013", "2023"])
        sub_df = comp_df[comp_df["Dataset"] == ds_sel].reset_index(drop=True)
        
        st.subheader(f"Model Comparison Table ({ds_sel} Dataset)")
        st.dataframe(sub_df.style.highlight_max(axis=0, subset=["Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"], color="#dbeafe"), use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Model Comparison Chart Across Datasets")
        metric_choice = st.selectbox("Select Metric to Visualize", ["Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"])
        
        fig_comp = create_model_comparison_bar(comp_df, metric_choice)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Best Model Calculation dynamically based on F1 Score
        best_row = comp_df.loc[comp_df["F1 Score"].idxmax()]
        
        st.markdown(f"""
            <div class='best-model-card'>
                <h2>🏆 Overall Best Model: {best_row['Model']} ({best_row['Dataset']} Dataset)</h2>
                <div style='display: flex; gap: 20px; flex-wrap: wrap; margin-top: 15px;'>
                    <div><strong>F1 Score:</strong> {best_row['F1 Score']:.4f}</div>
                    <div><strong>Precision:</strong> {best_row['Precision']:.4f}</div>
                    <div><strong>Recall:</strong> {best_row['Recall']:.4f}</div>
                    <div><strong>ROC-AUC:</strong> {best_row['ROC-AUC']:.4f}</div>
                    <div><strong>PR-AUC:</strong> {best_row['PR-AUC']:.4f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='warning-box'>
                ⚠️ <strong>Important Data Science Note:</strong><br>
                Do NOT rely solely on Accuracy for credit card fraud detection. Due to extreme class imbalance in financial transaction datasets 
                (such as 2013 where fraud is 0.17%), a dummy classifier predicting 'Legitimate' for every transaction achieves 99.83% accuracy while missing 100% of frauds! 
                Therefore, models must be evaluated using <strong>Recall, F1 Score, and PR-AUC</strong>.
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 4: FRAUD PREDICTION (AUTOMATIC SELECTION)
# ==========================================
elif nav_page == "Fraud Prediction":
    st.markdown("<h1>Interactive Fraud Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.05rem;'>Upload transaction data for <strong>Automatic Dataset & Model Selection</strong> based on schema compatibility, distribution similarity, and historical validation performance.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sample download expander
    with st.expander("📥 Need a sample CSV (2013, 2023, or 2015) to test automatic selection? Click to expand"):
        st.write("Generate or download sample transaction data:")
        samp_col1, samp_col2 = st.columns(2)
        with samp_col1:
            s13_df = load_dataset_sample("2013", sample_size=25)
            s13_bytes = s13_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download 2013 Sample CSV",
                data=s13_bytes,
                file_name="sample_transactions_2013.csv",
                mime="text/csv"
            )
        with samp_col2:
            s23_df = load_dataset_sample("2023", sample_size=25)
            s23_bytes = s23_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download 2023 Sample CSV",
                data=s23_bytes,
                file_name="sample_transactions_2023.csv",
                mime="text/csv"
            )
            
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Step 1: Upload Transaction CSV")
    uploaded_file = st.file_uploader("Upload a transaction CSV matching any card fraud schema:", type=["csv"])
    
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            meta = inspect_uploaded_file(uploaded_df)
            
            st.success(f"File uploaded successfully! Loaded {meta['rows']:,} rows and {meta['cols']} columns.")
            
            # Step 1 Metadata Inspection Display
            st.markdown("#### Uploaded File Metadata Summary")
            u1, u2, u3, u4, u5 = st.columns(5)
            u1.metric("Rows", f"{meta['rows']:,}")
            u2.metric("Columns", meta['cols'])
            u3.metric("Missing Values", meta['missing'])
            u4.metric("Duplicate Rows", f"{meta['duplicates']:,}")
            u5.metric("Has Class Col", "Yes" if "Class" in meta['columns'] else "No")
            
            with st.expander("Show Detailed Column Information & Preview"):
                st.write("**First 5 Rows:**")
                st.dataframe(uploaded_df.head(5), use_container_width=True)
                st.write("**Column Data Types:**")
                st.json(meta['dtypes'])
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Step 2 & 3: Automatic Dataset & Model Analysis")
            
            # Load reference statistics and sample models for evaluation
            ref_stats_13, ref_stats_23 = get_cached_reference_stats()
            sample_m13 = load_model("2013", "Random Forest")
            sample_m23 = load_model("2023", "Random Forest")
            
            # Run Automatic Selection Algorithm
            decision = evaluate_automatic_selection(
                uploaded_df,
                ref_stats_13,
                ref_stats_23,
                comp_df,
                {"2013": sample_m13, "2023": sample_m23}
            )
            
            if not decision["compatible"]:
                st.error(f"❌ {decision['error_msg']}")
            else:
                # Step 9 Visual Decision Card & Comparison Table
                st.markdown("""
                    <div style='background: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px;'>
                        <h4 style='color: #0f172a; margin-top: 0; text-align: center;'>AUTOMATIC MODEL SELECTION WORKFLOW</h4>
                        <div style='display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; text-align: center; font-size: 0.85rem; font-weight: 600; gap: 8px;'>
                            <div style='background: #f1f5f9; padding: 8px 12px; border-radius: 6px;'>Uploaded Dataset</div>
                            <div>➔</div>
                            <div style='background: #dbeafe; padding: 8px 12px; border-radius: 6px;'>Schema Compatibility</div>
                            <div>➔</div>
                            <div style='background: #dbeafe; padding: 8px 12px; border-radius: 6px;'>Distribution Similarity</div>
                            <div>➔</div>
                            <div style='background: #bfdbfe; padding: 8px 12px; border-radius: 6px;'>Dataset Selected</div>
                            <div>➔</div>
                            <div style='background: #93c5fd; padding: 8px 12px; border-radius: 6px;'>Historical Performance</div>
                            <div>➔</div>
                            <div style='background: #2563eb; color: white; padding: 8px 12px; border-radius: 6px;'>Best Model Selected</div>
                            <div>➔</div>
                            <div style='background: #1e40af; color: white; padding: 8px 12px; border-radius: 6px;'>Fraud Prediction</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Comparison Table Display
                st.write("**Dataset Family Evaluation Summary:**")
                st.table(decision["table_summary"])
                
                # Recommendation Highlight Box
                st.markdown(f"""
                    <div class='auto-selection-card'>
                        <h3 style='margin-top: 0;'>🤖 Automatic Selection Recommendation</h3>
                        <div style='display: flex; gap: 30px; flex-wrap: wrap; margin-bottom: 15px;'>
                            <div><strong>Selected Dataset:</strong> {decision['selected_dataset']} Dataset</div>
                            <div><strong>Selected ML Algorithm:</strong> {decision['selected_model']}</div>
                            <div><strong>Overall Compatibility Score:</strong> {decision['dataset_score']:.1f}%</div>
                            <div><strong>Model Selection Score:</strong> {decision['model_score']:.4f}</div>
                        </div>
                        <p style='margin-bottom: 5px;'>💬 <strong>Dataset Reason:</strong> {decision['dataset_explanation']}</p>
                        <p style='margin-bottom: 0;'>💬 <strong>Model Reason:</strong> {decision['model_explanation']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if decision.get("close_call_warning"):
                    st.warning(f"⚠️ {decision['close_call_warning']}")
                    
                # Step 8 Important Warning Box
                st.markdown("""
                    <div class='insight-card'>
                        ℹ️ <strong>Automatic Model Selection Disclaimer:</strong><br>
                        Automatic model selection is based on schema compatibility, feature-distribution similarity, and historical validation performance. 
                        Because uploaded prediction data normally has no ground-truth fraud labels, the system cannot guarantee that the selected model is objectively the most accurate for the new data.
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Step 4: Run Fraud Detection")
                
                # Execute Prediction
                if st.button("🚨 Run Fraud Detection"):
                    with st.spinner(f"Loading selected pre-trained model ({decision['selected_model']} - {decision['selected_dataset']}) and running predictions..."):
                        selected_pipeline = load_model(decision['selected_dataset'], decision['selected_model'])
                        results = run_fraud_prediction(selected_pipeline, uploaded_df, decision['selected_dataset'])
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.subheader("Prediction KPI Results")
                        
                        p1, p2, p3, p4 = st.columns(4)
                        p1.metric("Transactions Checked", f"{results['total_checked']:,}")
                        p2.metric("Fraud Flagged", f"{results['flagged_count']:,}")
                        p3.metric("Flagged Percentage", f"{results['flagged_pct']:.2f}%")
                        p4.metric("Highest Fraud Probability", f"{results['highest_prob']:.4f}")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.subheader("Flagged Fraudulent Transactions")
                        
                        flagged_df = results["flagged_df"]
                        if flagged_df.empty:
                            st.success("🎉 No fraudulent transactions flagged in this uploaded batch!")
                        else:
                            st.dataframe(
                                flagged_df.style.background_gradient(subset=["Fraud_Probability"], cmap="Reds"),
                                use_container_width=True
                            )
                            
                        if st.checkbox("Show All Predictions (Legitimate & Flagged)"):
                            st.dataframe(results["output_df"], use_container_width=True)
                            
                        st.markdown("<br>", unsafe_allow_html=True)
                        d_col1, d_col2 = st.columns(2)
                        with d_col1:
                            flagged_csv = flagged_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Flagged Transactions CSV",
                                data=flagged_csv,
                                file_name=f"flagged_transactions_{decision['selected_dataset']}_{decision['selected_model'].replace(' ', '_')}.csv",
                                mime="text/csv"
                            )
                        with d_col2:
                            all_csv = results["output_df"].to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download All Predictions CSV",
                                data=all_csv,
                                file_name=f"all_predictions_{decision['selected_dataset']}_{decision['selected_model'].replace(' ', '_')}.csv",
                                mime="text/csv"
                            )
                            
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.subheader("Fraud Probability Distribution Histogram")
                        fig_hist = create_probability_histogram(results["probabilities"])
                        st.plotly_chart(fig_hist, use_container_width=True)

        except Exception as e:
            st.error(f"Error analyzing or predicting on uploaded CSV file: {str(e)}")

# ==========================================
# PAGE 5: FEATURE IMPORTANCE
# ==========================================
elif nav_page == "Feature Importance":
    st.markdown("<h1>Feature Importance Analysis</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    fi_col1, fi_col2 = st.columns(2)
    with fi_col1:
        fi_year = st.selectbox("Select Dataset", ["2013", "2023"])
    with fi_col2:
        fi_model_name = st.selectbox("Select Model Algorithm", ["Random Forest", "XGBoost"])
        
    try:
        pipeline = load_model(fi_year, fi_model_name)
        imp_df = get_feature_importance_df(pipeline)
        
        if imp_df.empty:
            st.warning("Feature importance not available for selected model.")
        else:
            fig_fi = create_feature_importance_chart(imp_df, fi_year, fi_model_name)
            st.plotly_chart(fig_fi, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Top 15 Feature Importances Table")
            st.dataframe(imp_df.head(15), use_container_width=True)
            
            st.markdown("""
                <div class='insight-card'>
                    ℹ️ <strong>Data Science Note on Feature Importance:</strong><br>
                    Feature importance indicates how strongly a model relied on a feature during decision tree splits or predictions. 
                    It reflects predictive power within the dataset but does <strong>not imply direct causation</strong>.
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load model feature importance: {str(e)}")

# ==========================================
# PAGE 6: ABOUT PROJECT
# ==========================================
elif nav_page == "About Project":
    st.markdown("<h1>About Project & Technical Documentation</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Problem Statement
    Credit card fraud is difficult to detect because fraudulent transactions represent a tiny fraction of overall transaction volume. 
    Traditional accuracy-based evaluation can be severely misleading when evaluated under high class imbalance.

    ### 📌 Project Objective
    Develop and compare machine-learning fraud detection models using two independent datasets representing different time periods (2013 vs 2023).

    ### 🤖 Machine Learning Models Trained
    - **Logistic Regression**: Baseline linear classifier using class-weighted cost function.
    - **Random Forest**: Non-linear ensemble model using bagged decision trees.
    - **XGBoost**: Gradient boosted decision tree framework optimized for imbalanced classification.

    ### 🔄 Preprocessing Strategy
    - Stratified Train/Test split (80% Train, 20% Test) preserving exact class ratios.
    - `StandardScaler` applied to continuous columns (`Time`, `Amount`) inside Scikit-learn pipelines.
    - Independent preprocessing pipelines for 2013 and 2023 datasets.

    ### 📊 Evaluation Metrics
    - **Precision**: Proportion of flagged transactions that were actually fraudulent.
    - **Recall (Sensitivity)**: Proportion of actual fraudulent transactions successfully flagged.
    - **F1 Score**: Harmonic mean of Precision and Recall.
    - **ROC-AUC & PR-AUC**: Area under Receiver Operating Characteristic and Precision-Recall curves.

    ### 🔬 Main Research Question
    *"Do fraud detection patterns and model performance differ between the 2013 and 2023 datasets?"*
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Project Workflow & Architecture")
    
    st.markdown("""
        <div style='background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;'>
            <div style='display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; text-align: center; gap: 10px;'>
                <div style='background: #eff6ff; padding: 12px; border-radius: 8px; width: 140px; font-weight: 600;'>CSV Datasets<br><small>(2013 & 2023)</small></div>
                <div>➔</div>
                <div style='background: #eff6ff; padding: 12px; border-radius: 8px; width: 140px; font-weight: 600;'>EDA & Inspection</div>
                <div>➔</div>
                <div style='background: #eff6ff; padding: 12px; border-radius: 8px; width: 140px; font-weight: 600;'>Stratified Split</div>
                <div>➔</div>
                <div style='background: #eff6ff; padding: 12px; border-radius: 8px; width: 140px; font-weight: 600;'>Pipeline Preprocessing</div>
            </div>
            <div style='display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; text-align: center; gap: 10px; margin-top: 15px;'>
                <div style='background: #dbeafe; padding: 12px; border-radius: 8px; width: 140px; font-weight: 600;'>Model Training<br><small>(LR, RF, XGB)</small></div>
                <div>➔</div>
                <div style='background: #dbeafe; padding: 12px; border-radius: 8px; width: 140px; font-weight: 600;'>Metric Evaluation</div>
                <div>➔</div>
                <div style='background: #dbeafe; padding: 12px; border-radius: 8px; width: 140px; font-weight: 600;'>Saved PKL Models</div>
                <div>➔</div>
                <div style='background: #1e40af; color: white; padding: 12px; border-radius: 8px; width: 140px; font-weight: 600;'>Streamlit Dashboard</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
