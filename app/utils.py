import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def inject_custom_css():
    st.markdown("""
        <style>
        /* Main background & fonts */
        .main {
            background-color: #f8fafc;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0f172a !important;
            color: #f8fafc;
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        
        /* Header typography */
        h1, h2, h3 {
            color: #0f172a;
            font-weight: 700;
        }
        
        /* Modern Card Styling */
        .kpi-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            border: 1px solid #e2e8f0;
            text-align: center;
            transition: transform 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
        }
        .kpi-title {
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .kpi-value {
            font-size: 1.75rem;
            font-weight: 800;
            color: #0f172a;
        }
        .kpi-subtitle {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 4px;
        }
        
        .kpi-value-fraud {
            color: #dc2626;
        }
        .kpi-value-safe {
            color: #16a34a;
        }
        
        /* Highlight cards */
        .best-model-card {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white !important;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
            margin-bottom: 20px;
        }
        .best-model-card * {
            color: white !important;
        }
        
        .auto-selection-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white !important;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.4);
            margin: 20px 0;
            border: 1px solid #334155;
        }
        .auto-selection-card * {
            color: white !important;
        }
        
        .warning-box {
            background-color: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 16px;
            border-radius: 6px;
            margin: 16px 0;
            color: #991b1b;
        }
        
        .insight-card {
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 16px;
            border-radius: 6px;
            margin: 16px 0;
            color: #1e40af;
            font-weight: 500;
        }
        
        /* Table styling */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Button styling */
        .stButton button {
            background-color: #2563eb;
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            padding: 10px 20px;
        }
        .stButton button:hover {
            background-color: #1d4ed8;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_dataset(year: str) -> pd.DataFrame:
    filename = f"creditcard_{year}.csv"
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        st.error(f"Dataset file non-existent at path: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_dataset_sample(year: str, sample_size: int = 1000) -> pd.DataFrame:
    df = load_dataset(year)
    if df.empty:
        return df
    if len(df) > sample_size:
        return df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df

@st.cache_data
def load_results():
    comp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    cv_path = os.path.join(RESULTS_DIR, "cross_validation_results.csv")
    dist_path = os.path.join(RESULTS_DIR, "class_distribution_comparison.csv")
    
    comp_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else pd.DataFrame()
    cv_df = pd.read_csv(cv_path) if os.path.exists(cv_path) else pd.DataFrame()
    dist_df = pd.read_csv(dist_path) if os.path.exists(dist_path) else pd.DataFrame()
    
    return comp_df, cv_df, dist_df

@st.cache_resource
def load_model(year: str, model_name: str):
    name_map = {
        "Logistic Regression": f"logistic_regression_{year}.pkl",
        "Random Forest": f"random_forest_{year}.pkl",
        "XGBoost": f"xgboost_{year}.pkl"
    }
    filename = name_map.get(model_name)
    if not filename:
        raise ValueError(f"Unknown model name: {model_name}")
    
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    return joblib.load(path)

@st.cache_data
def get_cached_reference_stats():
    """
    Returns pre-computed reference distribution stats for 2013 and 2023 raw datasets.
    """
    from predictor import compute_reference_stats
    df_13 = load_dataset_sample("2013", sample_size=50000)
    df_23 = load_dataset_sample("2023", sample_size=50000)
    
    stats_13 = compute_reference_stats(df_13)
    stats_23 = compute_reference_stats(df_23)
    return stats_13, stats_23

# Chart creation functions
def create_class_distribution_pie(df: pd.DataFrame, year: str):
    if df.empty or "Class" not in df.columns:
        return go.Figure()
    
    counts = df["Class"].value_counts().reset_index()
    counts.columns = ["Class_Val", "Count"]
    counts["Label"] = counts["Class_Val"].map({0: "Legitimate", 1: "Fraud"})
    
    fig = px.pie(
        counts,
        values="Count",
        names="Label",
        title=f"Legitimate vs Fraud Transactions ({year})",
        hole=0.4,
        color="Label",
        color_discrete_map={"Legitimate": "#3b82f6", "Fraud": "#ef4444"}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig

def create_grouped_class_comparison(dist_df: pd.DataFrame):
    if dist_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dist_df["Dataset"],
        y=dist_df["Legitimate %"],
        name="Legitimate %",
        marker_color="#3b82f6",
        text=[f"{v:.2f}%" for v in dist_df["Legitimate %"]],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=dist_df["Dataset"],
        y=dist_df["Fraud %"],
        name="Fraud %",
        marker_color="#ef4444",
        text=[f"{v:.2f}%" for v in dist_df["Fraud %"]],
        textposition="auto"
    ))
    fig.update_layout(
        title="Class Distribution Comparison (% Legitimate vs % Fraud)",
        barmode="group",
        yaxis_title="Percentage (%)",
        xaxis_title="Dataset",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig

def create_amount_distribution_chart(df: pd.DataFrame, year: str):
    if df.empty or "Amount" not in df.columns or "Class" not in df.columns:
        return go.Figure()
    
    plot_df = df.sample(n=min(10000, len(df)), random_state=42).copy()
    plot_df["Transaction Class"] = plot_df["Class"].map({0: "Legitimate", 1: "Fraud"})
    
    fig = px.box(
        plot_df,
        x="Transaction Class",
        y="Amount",
        color="Transaction Class",
        color_discrete_map={"Legitimate": "#3b82f6", "Fraud": "#ef4444"},
        title=f"Transaction Amount Distribution by Class ({year})",
        points="outliers",
        log_y=True
    )
    fig.update_layout(
        yaxis_title="Amount (Log Scale)",
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig

def create_correlation_heatmap(df: pd.DataFrame, year: str):
    if df.empty:
        return go.Figure()
    
    num_df = df.select_dtypes(include=[np.number])
    cols_to_plot = [c for c in ["Amount", "Class", "V1", "V2", "V3", "V4", "V7", "V10", "V11", "V12", "V14", "V17"] if c in num_df.columns]
    if len(cols_to_plot) < 5:
        cols_to_plot = list(num_df.columns[:15])
        
    corr = num_df[cols_to_plot].corr()
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=f"Feature Correlation Heatmap ({year})"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_model_comparison_bar(comp_df: pd.DataFrame, metric: str):
    if comp_df.empty or metric not in comp_df.columns:
        return go.Figure()
    
    fig = px.bar(
        comp_df,
        x="Model",
        y=metric,
        color="Dataset",
        barmode="group",
        title=f"Model Comparison across Datasets ({metric})",
        text_auto=".4f",
        color_discrete_map={"2013": "#1e40af", "2023": "#3b82f6"}
    )
    fig.update_layout(
        yaxis_range=[0, 1.05],
        yaxis_title=metric,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig

def get_feature_importance_df(pipeline) -> pd.DataFrame:
    try:
        model = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessing"]
        
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [name.split("__")[-1] for name in feature_names]
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()
        
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        return imp_df
    except Exception:
        return pd.DataFrame()

def create_feature_importance_chart(imp_df: pd.DataFrame, year: str, model_name: str, top_n: int = 15):
    if imp_df.empty:
        return go.Figure()
    
    top_df = imp_df.head(top_n).sort_values(by="Importance", ascending=True)
    
    fig = px.bar(
        top_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Important Features — {model_name} ({year})",
        color="Importance",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False
    )
    return fig

def create_probability_histogram(probabilities: np.ndarray):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=probabilities,
        xbins=dict(start=0.0, end=1.0, size=0.1),
        marker_color="#3b82f6",
        marker_line_color="#1e40af",
        marker_line_width=1.5
    ))
    fig.update_layout(
        title="Fraud Probability Distribution",
        xaxis_title="Fraud Probability Range",
        yaxis_title="Transaction Count",
        xaxis=dict(tickmode="array", tickvals=[i/10 for i in range(11)]),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
